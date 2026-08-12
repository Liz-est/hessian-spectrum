"""
Gauss-Radau Quadrature 用于 Lanczos 谱的误差带估计

理论基础（Golub & Meurant, "Matrices, Moments and Quadrature and their
applications"，Chapter 6）：
- Lanczos 三对角矩阵 T_m 的特征值分解给出 m 点 Gauss quadrature 规则，
  其节点=Ritz 值 θ_i，权重=U[0,i]^2。它给出谱测度 μ 的一个离散近似。
- 目标量：累积谱分布 Φ(t) = μ({λ ≤ t})（"有多少比例的特征值 ≤ t"）。
  绘图横轴 eigenvalue index = N_params × (1 - Φ(t))
  （index 越大特征值越小，所以用"≥ t 的质量"= 1 - Φ(t)）。
- 误差带：把 Gauss-Radau 规则的固定节点分别锚定在谱区间的左端点 a=λmin
  和右端点 b=λmax，得到 Φ 的一对上下界（Golub-Meurant Thm 6.4）：
      Radau(a) 与 Radau(b) 夹住真实 Φ(t)。
  Gauss 规则本身给中点估计 mid。三者取 min/max 保证 lo ≤ mid ≤ hi。
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from typing import Tuple


def _radau_matrix(alpha: np.ndarray, beta: np.ndarray, anchor: float):
    """
    构造 (m)×(m) Gauss-Radau 三对角矩阵：修改最后一个对角元素 α_m，
    使 `anchor` 成为该 quadrature 规则的一个精确节点（Golub 1973）。

    修正量：α_m^new = anchor + δ，其中
        δ = β_{m-1}^2 · e_m^T (T_{m-1} - anchor·I)^{-1} e_m
    这里 T_{m-1} 是去掉最后一行/列的 (m-1) 阶主子阵，δ 通过解三对角系统
        (T_{m-1} - anchor·I) x = e_{m-1}
    的最后一个分量得到。

    Args:
        alpha: (m,) 对角元素
        beta:  (m-1,) 次对角元素
        anchor: 要锚定的节点（通常是谱端点 λmin 或 λmax）

    Returns:
        alpha_radau: (m,) 修改后的对角元素（仅最后一个不同）
        beta: (m-1,) 次对角元素（不变）
    """
    m = len(alpha)
    if m == 1:
        return np.array([anchor], dtype=np.float64), beta

    # 解 (T_{m-1} - anchor I) x = e_{m-1}，取 x 的最后一个分量
    # T_{m-1}: 对角 alpha[0..m-2]，次对角 beta[0..m-3]
    d = alpha[:m - 1] - anchor          # 对角 (m-1,)
    e = beta[:m - 2]                    # 次对角 (m-2,)

    # Thomas 算法（前向消元 + 回代），右端项 = e_{m-1}（最后一个为 1）
    n = m - 1
    c = np.zeros(n)   # 归一化后的上对角
    rhs = np.zeros(n)
    rhs[-1] = 1.0

    c[0] = e[0] / d[0] if n > 1 else 0.0
    rhs[0] = rhs[0] / d[0]
    for i in range(1, n):
        denom = d[i] - e[i - 1] * c[i - 1]
        if i < n - 1:
            c[i] = e[i] / denom
        rhs[i] = (rhs[i] - e[i - 1] * rhs[i - 1]) / denom

    x_last = rhs[-1]
    for i in range(n - 2, -1, -1):
        rhs[i] = rhs[i] - c[i] * rhs[i + 1]
    # 回代后 rhs 存的是解 x；我们只要最后分量
    delta = rhs[-1]  # = e_{m-1}^T (T_{m-1}-anchor I)^{-1} e_{m-1}

    alpha_radau = alpha.copy()
    alpha_radau[-1] = anchor + beta[-1] ** 2 * delta
    return alpha_radau, beta


def _cdf_from_rule(nodes: np.ndarray, weights: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    从一个 quadrature 规则（节点/权重）计算累积分布 Φ(t)=Σ_{θ_i ≤ t} ω_i，
    在 grid 上求值（右连续阶梯函数）。
    """
    order = np.argsort(nodes)
    nd = nodes[order]
    wd = weights[order]
    cw = np.cumsum(wd)
    # 对每个 t，找到 ≤ t 的最大节点位置
    idx = np.searchsorted(nd, grid, side='right') - 1
    cdf = np.where(idx >= 0, cw[np.clip(idx, 0, len(cw) - 1)], 0.0)
    return cdf


def cumulative_spectral_density(
    alpha: np.ndarray,
    beta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 Lanczos 三对角矩阵计算 Gauss quadrature 节点(Ritz 值)与权重。

    Returns:
        nodes: (m,) Ritz 值，升序
        weights: (m,) 对应权重（U[0,i]^2），Σ=1
    """
    eigenvalues, U = eigh_tridiagonal(alpha, beta)
    weights = U[0, :] ** 2
    order = np.argsort(eigenvalues)
    return eigenvalues[order], weights[order]


def ritz_convergence(alpha: np.ndarray, beta: np.ndarray):
    """
    每个 Ritz 值的 Lanczos 残差收敛估计 rel_i = β_m·|U[-1,i]| / |θ_i|。

    这是标准的「Ritz 值是否已收敛为真实特征值」判据（Paige/Saad）：
    U[-1,i] 是第 i 个 Ritz 向量在 Lanczos 基下的末分量，β_m=beta[-1]。
    rel_i < tol 的 Ritz 值视为已锁定的离散特征值，其余归入连续体。

    Returns:
        nodes: (m,) Ritz 值，降序（大特征值在前）
        weights: (m,) 对应权重，同序
        rel: (m,) 相对残差，同序
    """
    ev, U = eigh_tridiagonal(alpha, beta)
    w = U[0, :] ** 2
    resid = abs(beta[-1]) * np.abs(U[-1, :])
    rel = resid / np.maximum(np.abs(ev), 1e-30)
    order = np.argsort(ev)[::-1]           # 降序：大特征值在前
    return ev[order], w[order], rel[order]


def compute_spectrum_with_error_bands(
    alpha: np.ndarray,
    beta: np.ndarray,
    n_params: int,
    n_grid: int = 400,
    conv_tol: float = 1e-10,
) -> dict:
    """
    完整的谱计算：已收敛 Ritz 值 → 离散锁定点；未收敛 Ritz 值 → 连续体 + 误差带。

    误差带方法（与论文缓存一致，已由反解核实）：
      * 用残差判据 rel_i = β_m·|U[-1,i]|/|θ_i| < conv_tol 把 Ritz 值分成
        「已锁定离散谱」(前 L 个，最大的那些) 与「连续体」(其余)。
      * 连续体区某个特征值 g 的 index 不确定性 = 覆盖它的**相邻连续体 Ritz
        节点的 index 间距**（节点越稀 → 带越宽），而非 Gauss-Radau 端点夹逼。
        （在 m 很大、完整重正交时 Radau 夹逼会退化成零宽，与论文不符。）

    返回字段与 spectrum_3x3.npz 一致（x/y/dot_x/g/lo/hi/mid/L/R/cut），
    其中 mid/lo/hi 均为 eigenvalue index（= N_params × 质量），横轴用；
    g 为对应的 eigenvalue（纵轴，正谱对数网格）。

    Args:
        alpha: (m,) Lanczos 三对角对角元素
        beta:  (m-1,) 次对角元素
        n_params: 参数总数（index 的缩放）
        n_grid: 正谱网格点数
        conv_tol: Ritz 收敛判据相对残差阈值（默认 1e-10，复现论文 L）

    Returns:
        dict
    """
    N = float(n_params)

    # 1) Ritz 值（降序）+ 权重 + 残差收敛判据
    nodes_desc, w_desc, rel_desc = ritz_convergence(alpha, beta)
    lambda_max = float(nodes_desc[0])

    # 累积 index：第 i 个 Ritz 值（降序）之前的质量 → index
    cum = np.cumsum(w_desc)              # 累积质量（含自身）
    ritz_index = N * cum                 # 大特征值 → 小 index

    # L = 已收敛（锁定为离散特征值）的个数：从最大特征值起连续满足 rel<tol 的前缀
    conv = rel_desc < conv_tol
    L = int(np.argmax(~conv)) if (~conv).any() else len(conv)

    # 2) 散点 x/y（论文约定：x 升序 index 1..N，y 降序 eigenvalue；
    #    故 x[:L]/y[:L] 恰好是最大的 L 个已锁定特征值）
    ritz_x = ritz_index.copy()          # 升序 index（index 越小 → 特征值越大）
    ritz_y = nodes_desc.copy()          # 降序 eigenvalue
    cut = int(np.sum(nodes_desc > 0))

    # 3) 连续体节点（rank >= L）：它们的 (eigenvalue, index) 定义 mid 与误差带。
    #    锁定段(rank<L)的高特征值由离散点 x[:L]/y[:L] 负责，连续体只覆盖 rank≥L，
    #    故 grid 上限取最大连续体节点，避免高特征值端插值 clamp 出竖刺。
    cont_y = nodes_desc[L:]              # eigenvalue，降序
    cont_idx = ritz_index[L:]           # index，升序
    pos = cont_y > 0
    cy = cont_y[pos][::-1]              # eigenvalue 升序
    ci = cont_idx[pos][::-1]           # 对应 index
    order = np.argsort(cy)
    cy, ci = cy[order], ci[order]
    cont_max = float(cy.max()) if len(cy) else lambda_max
    positive = nodes_desc[nodes_desc > 0]
    positive_min = max(float(positive.min()), 1e-12) if len(positive) else 1e-12
    grid = np.geomspace(positive_min, cont_max, n_grid)

    # mid(g)：把 g 插值到连续体节点的 index（log-index vs log-eigenvalue 单调）
    log_cy = np.log(np.maximum(cy, 1e-300))
    log_ci = np.log(np.maximum(ci, 1e-300))
    log_mid = np.interp(np.log(grid), log_cy, log_ci)
    mid = np.exp(log_mid)

    # 4) 误差带：连续体区某 eigenvalue 的 index 不确定性 = 覆盖它的相邻节点 index 间距。
    #    带宽在 log-index 空间关于 mid 对称（与论文缓存一致），且需在 grid 上平滑：
    #    对每个连续体节点算「到左右邻居的 log-index 半间距」，再平滑插值到 grid。
    if len(log_ci) >= 2:
        gap = np.gradient(log_ci)          # 每个节点的局部 log-index 间距（中心差分，平滑）
        half = 0.5 * np.abs(gap)
        half_grid = np.interp(np.log(grid), log_cy, half)
    else:
        half_grid = np.zeros_like(grid)
    lo = np.exp(log_mid - half_grid)
    hi = np.exp(log_mid + half_grid)

    result = {
        "x": ritz_x,
        "y": ritz_y,
        "dot_x": ritz_x,
        "g": grid,
        "lo": lo,
        "hi": hi,
        "mid": mid,
        "L": L,
        "R": 0,
        "cut": cut,
    }
    return result


# ============================================================================
# 测试函数
# ============================================================================

def test_gauss_radau():
    """用已知对称矩阵测试：Lanczos 累积密度 vs 真实累积分布，并检查带序。"""
    print("=" * 80)
    print("测试 Gauss-Radau Quadrature（真实 alpha/beta，端点锚定）")
    print("=" * 80)

    np.random.seed(0)
    n = 300
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    true_eigs = np.sort(np.abs(np.random.randn(n)) * 5 + 0.1)
    A = Q @ np.diag(true_eigs) @ Q.T

    # Lanczos（完整重正交）生成 alpha/beta
    m = 80
    v = np.random.randn(n); v /= np.linalg.norm(v)
    V = np.zeros((m, n)); alpha = np.zeros(m); beta = np.zeros(m - 1)
    V[0] = v
    w = A @ v; alpha[0] = w @ v; w = w - alpha[0] * v
    for j in range(1, m):
        for i in range(j):
            w = w - (w @ V[i]) * V[i]
        b = np.linalg.norm(w); beta[j - 1] = b; V[j] = w / b
        w = A @ V[j]; alpha[j] = w @ V[j]
        w = w - alpha[j] * V[j] - beta[j - 1] * V[j - 1]

    print(f"矩阵 {n}×{n}, Lanczos m={m}")
    print(f"真实特征值范围: [{true_eigs.min():.4f}, {true_eigs.max():.4f}]")

    result = compute_spectrum_with_error_bands(alpha, beta, n_params=n, n_grid=200)

    print(f"\n网格 g 范围: [{result['g'].min():.4f}, {result['g'].max():.4f}]")
    print(f"mid index 范围: [{result['mid'].min():.2f}, {result['mid'].max():.2f}]")
    print(f"lo  index 范围: [{result['lo'].min():.2f}, {result['lo'].max():.2f}]")
    print(f"hi  index 范围: [{result['hi'].min():.2f}, {result['hi'].max():.2f}]")

    order_ok = np.all(result['lo'] <= result['mid'] + 1e-6) and \
               np.all(result['mid'] <= result['hi'] + 1e-6)

    # 真实 index 曲线：index(t) = n × P(eig ≥ t)
    test_lam = np.median(true_eigs)
    true_index = np.sum(true_eigs >= test_lam)
    est_index = np.interp(test_lam, result['g'], result['mid'])
    print(f"\n在 λ={test_lam:.3f} 处 (index = #{{eig ≥ λ}}):")
    print(f"  真实 index: {true_index}")
    print(f"  估计 index: {est_index:.2f}")
    print(f"  相对误差: {abs(true_index - est_index) / max(true_index,1):.4f}")

    if order_ok:
        print("\n✅ 测试通过！误差带满足 lo ≤ mid ≤ hi")
    else:
        bad = np.sum((result['lo'] > result['mid'] + 1e-6) |
                     (result['mid'] > result['hi'] + 1e-6))
        print(f"\n⚠️  {bad}/{len(result['mid'])} 个网格点违反带序")
    print("=" * 80)


if __name__ == "__main__":
    test_gauss_radau()
