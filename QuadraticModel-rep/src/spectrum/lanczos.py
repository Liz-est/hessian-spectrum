"""
Lanczos Algorithm 严格按照论文 Algorithm 1 实现
关键区别：重正交化在归一化之前（不是之后）
"""
import torch
import numpy as np
from typing import Callable, Tuple, Optional


def lanczos_algorithm_1(
    hvp_fn: Callable[[torch.Tensor], torch.Tensor],
    n_params: int,
    m: int,
    device: torch.device,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    store_device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    论文 Algorithm 1: Lanczos Quadrature with Full Reorthogonalization

    关键实现细节（与 toy_models 的区别）：
    1. 重正交化在归一化**之前**（论文第 6-7 行）
    2. 使用完整重正交化（对所有之前的向量）
    3. Lanczos 向量存储为 fp32，三对角矩阵为 fp64

    内存策略（大模型必需）：
    - 基向量矩阵 Q(m × n_params) 极大：167M 参数 × m=400 × fp32 = 268 GB，
      任何单 GPU 显存都装不下。因此 **Q 存于 `store_device`（默认 CPU 内存）**，
      每步仅把当前向量 q_j 搬到 `device`（GPU）做 HVP，结果搬回 CPU。
    - 重正交化用向量化 GEMV（DGKS：两轮经典 Gram-Schmidt，稳定性等价 MGS），
      而非逐向量 Python 循环（后者对 167M 维 × m 步会慢到不可用）。

    Args:
        hvp_fn: Hessian-vector product 函数，输入/输出都是扁平化的向量（在 device 上）
        n_params: 参数总数
        m: Lanczos 迭代深度
        device: HVP 计算设备（GPU）
        seed: 随机种子
        dtype: Lanczos 向量的精度
        store_device: Q 基向量存储设备（默认 CPU，避免 GPU OOM）

    Returns:
        eigenvalues: (m,) 的 Ritz 值（特征值估计）
        weights: (m,) 的 Ritz 权重（U[0]^2，用于绘图）
        alpha: (m,) 三对角矩阵对角元素（供 Gauss-Radau 使用）
        beta: (m-1,) 三对角矩阵次对角元素（供 Gauss-Radau 使用）
    """
    if store_device is None:
        store_device = device
    store_device = torch.device(store_device)
    device = torch.device(device)

    # 初始化随机向量 v0（在存储设备上生成并归一化）
    torch.manual_seed(seed)
    v = torch.randn(n_params, dtype=dtype, device=store_device)
    v = v / v.norm()

    # 存储 Lanczos 基向量 Q（fp32，存于 store_device / CPU）
    Q = torch.zeros(m, n_params, dtype=dtype, device=store_device)
    Q[0] = v

    # 三对角矩阵 T（fp64 for numerical stability）
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m - 1, dtype=np.float64)

    # Lanczos 迭代
    for j in range(m):
        # 计算 w = H * v_j（搬到 GPU 做 HVP，结果搬回存储设备）
        qj_dev = Q[j].to(device, non_blocking=True)
        w = hvp_fn(qj_dev).to(store_device)
        del qj_dev

        # 计算对角元素 α_j = <w, v_j>
        alpha[j] = torch.dot(w, Q[j]).item()

        # 完整重正交化（DGKS：两轮经典 Gram-Schmidt，向量化 GEMV）
        # 论文 Algorithm 1 第 6-7 行：z ← z - Σ <z, q_i> q_i
        # 这一步同时吸收了 α_j q_j 与 β_{j-1} q_{j-1} 的三项递推减法。
        Qv = Q[: j + 1]  # (j+1, n) 视图
        for _ in range(2):
            coeff = torch.mv(Qv, w)          # (j+1,) = Q · w
            w = w - torch.mv(Qv.t(), coeff)  # w -= Qᵀ (Q · w)

        # 归一化（在重正交化之后）
        # 论文 Algorithm 1 第 8 行
        beta_j = w.norm().item()

        # 检查提前终止
        if beta_j < 1e-10:
            print(f"Lanczos terminated early at iteration {j+1}/{m} (beta={beta_j:.2e})")
            # 截断矩阵
            alpha = alpha[:j + 1]
            beta = beta[:j]
            Q = Q[:j + 1]
            m = j + 1
            break

        # 存储 β_j 并归一化
        if j < m - 1:
            beta[j] = beta_j
            Q[j + 1] = w / beta_j
        if (j + 1) % 25 == 0 or j == m - 1:
            print(f"    Lanczos {j+1}/{m}  α={alpha[j]:.3e} β={beta_j:.3e}", flush=True)

    # 构造三对角矩阵 T
    T = np.diag(alpha)
    if len(beta) > 0:
        T[np.arange(m - 1), np.arange(1, m)] = beta
        T[np.arange(1, m), np.arange(m - 1)] = beta

    # 特征分解 T = U Λ U^T
    eigenvalues, U = np.linalg.eigh(T)

    # 计算 Ritz 权重（论文 Algorithm 1 第 11 行）
    # ω_i = (u_i^(1))^2，其中 u_i^(1) 是第 i 个特征向量的第一个分量
    weights = U[0, :] ** 2

    # 同时返回三对角矩阵的 alpha/beta（供 Gauss-Radau 使用）
    return eigenvalues, weights, alpha, beta


def compute_eigenvalue_index(eigenvalues: np.ndarray, weights: np.ndarray,
                            n_params: int) -> np.ndarray:
    """
    计算 eigenvalue index（用于绘图的横轴）

    论文方法：i_j = p * Σ_{k≤j} ω_k
    其中 p 是参数总数，ω_k 是 Ritz 权重

    Args:
        eigenvalues: (m,) Ritz 值
        weights: (m,) Ritz 权重
        n_params: 参数总数

    Returns:
        indices: (m,) eigenvalue index
    """
    # 按特征值从小到大排序
    order = np.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[order]
    weights_sorted = weights[order]

    # 计算累积权重和
    cumulative_weights = np.cumsum(weights_sorted)

    # 计算 index
    indices = n_params * cumulative_weights

    return indices, eigenvalues_sorted


def double_reorthogonalization(
    z: torch.Tensor,
    Q: torch.Tensor,
    j: int
) -> torch.Tensor:
    """
    双重 Gram-Schmidt 重正交化（备选实现，更稳定）

    论文建议：对数值不稳定的情况，可以重复重正交化两次

    Args:
        z: 待正交化的向量
        Q: (j+1, n) 已有的正交基
        j: 当前迭代索引

    Returns:
        z: 正交化后的向量
    """
    for _ in range(2):  # 重复两次
        for i in range(j + 1):
            z = z - torch.dot(z, Q[i]) * Q[i]
    return z


# ============================================================================
# 测试函数：用简单的矩阵验证实现正确性
# ============================================================================

def test_lanczos_on_known_matrix():
    """
    用已知特征值的矩阵测试 Lanczos 算法
    """
    print("=" * 80)
    print("测试 Lanczos Algorithm 1 实现")
    print("=" * 80)

    # 构造一个简单的对称矩阵
    n = 100
    device = torch.device("cpu")

    # 创建一个对角占优的对称矩阵
    A = torch.randn(n, n, dtype=torch.float32, device=device)
    A = (A + A.T) / 2  # 对称化
    A = A + 10 * torch.eye(n, device=device)  # 对角占优

    # 真实的特征值
    true_eigenvalues, _ = torch.linalg.eigh(A)
    true_eigenvalues = true_eigenvalues.cpu().numpy()

    print(f"矩阵大小: {n}×{n}")
    print(f"真实特征值范围: [{true_eigenvalues.min():.4f}, {true_eigenvalues.max():.4f}]")

    # 定义 HVP 函数
    def hvp_fn(v):
        return A @ v

    # 运行 Lanczos
    m = 50
    eigenvalues, weights, alpha, beta = lanczos_algorithm_1(
        hvp_fn=hvp_fn,
        n_params=n,
        m=m,
        device=device,
        seed=42,
    )

    print(f"\nLanczos 深度: m={m}")
    print(f"Ritz 值范围: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
    print(f"权重和: {weights.sum():.6f} (应该 ≈ 1.0)")

    # 比较最大和最小特征值
    print(f"\n特征值对比:")
    print(f"  最小特征值: 真实={true_eigenvalues[0]:.4f}, Lanczos={eigenvalues[0]:.4f}")
    print(f"  最大特征值: 真实={true_eigenvalues[-1]:.4f}, Lanczos={eigenvalues[-1]:.4f}")

    # 计算相对误差
    rel_err_min = abs(eigenvalues[0] - true_eigenvalues[0]) / abs(true_eigenvalues[0])
    rel_err_max = abs(eigenvalues[-1] - true_eigenvalues[-1]) / abs(true_eigenvalues[-1])
    print(f"\n相对误差:")
    print(f"  最小特征值: {rel_err_min:.2e}")
    print(f"  最大特征值: {rel_err_max:.2e}")

    if rel_err_min < 1e-3 and rel_err_max < 1e-3:
        print("\n✅ 测试通过！Lanczos 算法实现正确")
    else:
        print("\n⚠️  相对误差较大，可能需要增加 Lanczos 深度")

    print("=" * 80)

    return eigenvalues, weights


if __name__ == "__main__":
    # 运行测试
    test_lanczos_on_known_matrix()
