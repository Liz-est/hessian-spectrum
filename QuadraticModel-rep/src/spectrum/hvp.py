"""
Hessian-Vector Product (HVP) 和 Gauss-Newton-Vector Product 实现
支持 4 种类型：Raw/Preconditioned × GN/Hessian
"""
import torch
import torch.nn.functional as F
from typing import Callable, Optional, Dict

# 二阶导数（double backward）需要 math backend 的 attention。
# flash / efficient / cudnn 的 CUDA kernel 都没有实现二阶反向，
# 因此在计算 HVP 时用 sdpa_kernel 强制 math 实现（前向仍可用 flash）。
from torch.nn.attention import sdpa_kernel, SDPBackend

_MATH_SDPA = [SDPBackend.MATH]


def gauss_newton_vector_product(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    preconditioner: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Gauss-Newton 矩阵-向量乘积

    G = ∇²L_prox，Hessian 的 PSD 部分（去掉负曲率）

    实现基于 JVP + centered cotangent + transpose，参考：
    QuadraticModel/preprocessing/sample_hessian_frob2.py:220-256

    Args:
        model: PyTorch 模型
        x: 输入 tokens (B, T)
        y: 目标 tokens (B, T)
        v: 待乘向量（扁平化的参数向量）
        preconditioner: Adam 预条件器 P = diag[√ν + ε]（可选）

    Returns:
        Gv 或 P⁻¹GPv（如果提供预条件器）
    """
    # 1. 前向传播获取 logits（强制 math backend 以支持二阶导数）
    # 注意：某些模型（vanilla_model）在 targets=None 时只返回最后一个位置的
    # logits（推理优化），传入 targets 才返回完整序列 logits。这里传 y 以拿到
    # 完整序列，但损失仍由本函数自行计算（保持 loss 语义一致）。
    with sdpa_kernel(_MATH_SDPA):
        try:
            output = model(x, y)
        except TypeError:
            output = model(x)
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    # 2. 计算 softmax 概率
    q = F.softmax(logits, dim=-1)  # (B, T, V)

    # 3. 应用左预条件（如果提供）
    if preconditioner is not None:
        v_precond = apply_preconditioner(v, preconditioner, model)
    else:
        v_precond = v

    # 4. 将扁平化向量转换回参数结构
    v_dict = flat_to_params(v_precond, model)

    # 5. JVP: 计算 logits 对参数的 Jacobian-vector product  jvp = J·v
    #    正确做法用「双重反向 u-trick」（等价 jax.linearize 的 jvp_fn）：
    #      gu = ∂(<u, logits>)/∂θ = Jᵀu        （对 u 建图）
    #      inner = <gu, v> = uᵀ J v
    #      J·v = ∂inner/∂u                      （形状同 logits）
    #    ⚠ 旧实现用 grad_outputs=ones 后 torch.sum 把 JVP 塌成标量常数，
    #      经中心化后 GN 恒为 0（已被 diag_hvp.py 暴力对拍证伪）。
    params = list(model.parameters())
    u = torch.zeros_like(logits, requires_grad=True)
    gu = torch.autograd.grad(
        outputs=logits,
        inputs=params,
        grad_outputs=u,
        create_graph=True,
        retain_graph=True,
    )
    inner = torch.zeros((), device=logits.device, dtype=logits.dtype)
    for g_param, v_param in zip(gu, v_dict.values()):
        if g_param is not None and v_param is not None:
            inner = inner + torch.sum(g_param * v_param)
    (jvp_logits,) = torch.autograd.grad(
        outputs=inner,
        inputs=u,
        retain_graph=True,
    )  # (B, T, V) = J·v

    # 6. 中心化：jvp_centered = jvp - <q, jvp>
    # 这对应于去除 softmax 的零空间分量
    mean_jvp = torch.sum(q * jvp_logits, dim=-1, keepdim=True)
    jvp_centered = jvp_logits - mean_jvp  # (B, T, V)

    # 7. 计算 cotangent：q * centered / token_count
    token_count = y.numel()
    cotangent = q * jvp_centered / token_count  # (B, T, V)

    # 8. VJP (transpose): 计算梯度的转置
    # 这给出 G·v
    gv_params = torch.autograd.grad(
        outputs=logits,
        inputs=params,
        grad_outputs=cotangent,
        retain_graph=True,
    )

    # 9. 扁平化结果
    gv_flat = params_to_flat(gv_params, model)

    # 10. 应用右预条件（如果提供）
    if preconditioner is not None:
        gv_flat = apply_preconditioner(gv_flat, preconditioner, model)

    return gv_flat


def hessian_vector_product(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    preconditioner: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Hessian 矩阵-向量乘积

    H = ∇²L，包含所有二阶信息（正负曲率）

    使用 double backward 实现

    Args:
        model: PyTorch 模型
        x: 输入 tokens (B, T)
        y: 目标 tokens (B, T)
        v: 待乘向量（扁平化）
        preconditioner: Adam 预条件器（可选）

    Returns:
        Hv 或 P⁻¹HPv（如果提供预条件器）
    """
    # 1. 应用左预条件
    if preconditioner is not None:
        v_precond = apply_preconditioner(v, preconditioner, model)
    else:
        v_precond = v

    # 2. 前向传播（强制 math backend 以支持二阶导数）
    # 传入 y 以获取完整序列 logits（见 gauss_newton_vector_product 注释）
    with sdpa_kernel(_MATH_SDPA):
        try:
            output = model(x, y)
        except TypeError:
            output = model(x)
    # 处理模型可能返回元组的情况
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    # 3. 计算损失
    logits_flat = logits.reshape(-1, logits.size(-1))  # (B*T, V)
    y_flat = y.reshape(-1)  # (B*T,)
    loss = F.cross_entropy(logits_flat, y_flat, reduction='mean')

    # 4. 计算梯度
    params = list(model.parameters())
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=params,
        create_graph=True,
        retain_graph=True,
    )

    # 5. 将 v 转换为参数结构
    v_dict = flat_to_params(v_precond, model)

    # 6. 计算 <grad, v>（标量）
    grad_v = torch.tensor(0.0, device=v.device)
    for grad, v_param in zip(grads, v_dict.values()):
        if grad is not None and v_param is not None:
            grad_v += torch.sum(grad * v_param)

    # 7. Double backward: ∂(<grad, v>)/∂θ = H·v
    hv_params = torch.autograd.grad(
        outputs=grad_v,
        inputs=params,
        retain_graph=True,
    )

    # 8. 扁平化结果
    hv_flat = params_to_flat(hv_params, model)

    # 9. 应用右预条件
    if preconditioner is not None:
        hv_flat = apply_preconditioner(hv_flat, preconditioner, model)

    return hv_flat


def apply_preconditioner(
    v: torch.Tensor,
    preconditioner: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> torch.Tensor:
    """应用预条件器（按 param 施加），支持两种口径：

    1. **对角**（Adam / CompleteP-raw）：preconditioner[name] 是与 param 同形张量 →
       逐元素乘 v_dict[name] * P[name]。

    2. **dense-op**（Muon Kronecker）：preconditioner[name] 是 dict
       {"Phalf": (L,d,d), "side": "left"|"right"}，把 v_dict[name] 按 muon_reshape 视作
       (L,m,n) 逐层矩阵，施加对称半预条件器 Phalf：
         side="right"(tall m≥n): v_mat @ Phalf    （Phalf 为 n×n）
         side="left" (wide m<n): Phalf @ v_mat    （Phalf 为 m×m）
       两侧（HVP 的 v 入口 + gv 出口）各调用一次 → 得 𝒫^{-1/2} H 𝒫^{-1/2}。
       √(pre·post) 标量已并入 Phalf。

    未在 preconditioner 里的 param 保持不变。
    """
    from src.optim.opt import muon_reshape   # 逐层 2D reshape 约定（与优化器一致），避免循环 import

    v_dict = flat_to_params(v, model)
    v_precond_dict = {}
    for name, param in model.named_parameters():
        if name not in v_dict:
            continue
        P = preconditioner.get(name) if preconditioner else None
        if P is None:
            v_precond_dict[name] = v_dict[name]
        elif isinstance(P, dict):
            # dense-op（Muon）：reshape → bmm → reshape 回原形
            vm = muon_reshape(name, v_dict[name])        # (L,m,n)
            Phalf = P["Phalf"].to(vm.dtype)
            if P["side"] == "right":
                out = vm @ Phalf                          # (L,m,n)@(L,n,n)
            else:
                out = Phalf @ vm                          # (L,m,m)@(L,m,n)
            v_precond_dict[name] = out.reshape(param.shape)
        else:
            # 对角：逐元素乘
            v_precond_dict[name] = v_dict[name] * P

    return params_to_flat(v_precond_dict.values(), model)


def build_adam_preconditioner(
    nu: Dict[str, torch.Tensor],
    lr: Dict[str, float],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    构建 Adam 预条件器

    P = diag[√(lr_pre * lr_post / (√ν + ε))]

    参考：QuadraticModel/preprocessing/sample_hessian_frob2.py:230-238

    Args:
        nu: Adam 二阶矩估计（ν）
        lr: 学习率字典 {param_name: lr_value}
        eps: Adam epsilon

    Returns:
        preconditioner: {param_name: √(lr / (√ν + ε))}
    """
    preconditioner = {}

    for name, nu_param in nu.items():
        if name in lr:
            lr_value = lr[name]
            # P = √(lr / (√ν + ε))
            preconditioner[name] = torch.sqrt(
                lr_value / (torch.sqrt(nu_param) + eps)
            )
        else:
            # 如果没有 lr，使用单位预条件器
            preconditioner[name] = torch.ones_like(nu_param)

    return preconditioner


def params_to_flat(params, model: torch.nn.Module) -> torch.Tensor:
    """
    将参数列表扁平化为单个向量

    Args:
        params: 参数列表或字典值
        model: 模型（用于获取设备）

    Returns:
        扁平化向量
    """
    if isinstance(params, dict):
        params = params.values()

    flat_params = []
    for param in params:
        if param is not None:
            flat_params.append(param.reshape(-1))

    if len(flat_params) == 0:
        # 返回空向量
        device = next(model.parameters()).device
        return torch.tensor([], device=device)

    return torch.cat(flat_params)


def flat_to_params(
    flat_vec: torch.Tensor,
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    将扁平化向量转换回参数结构

    Args:
        flat_vec: 扁平化向量
        model: 模型

    Returns:
        参数字典 {name: param_tensor}
    """
    params_dict = {}
    offset = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        param_flat = flat_vec[offset:offset + numel]
        params_dict[name] = param_flat.reshape(param.shape)
        offset += numel

    return params_dict


# ============================================================================
# 测试函数
# ============================================================================

def test_hvp_on_toy_model():
    """
    在简单模型上测试 HVP
    """
    print("=" * 80)
    print("测试 HVP 和 GN-VP 实现")
    print("=" * 80)

    # 创建一个简单的线性模型
    class ToyModel(torch.nn.Module):
        def __init__(self, vocab_size=100, hidden_dim=50):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
            self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

        def forward(self, x):
            h = self.embedding(x)  # (B, T, D)
            logits = self.lm_head(h)  # (B, T, V)
            return logits

    model = ToyModel()
    model.eval()

    # 创建测试数据
    B, T, V = 4, 10, 100
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))

    # 创建随机向量
    n_params = sum(p.numel() for p in model.parameters())
    v = torch.randn(n_params)

    print(f"模型参数数量: {n_params}")
    print(f"输入形状: {x.shape}")

    # 测试 GN-VP
    print("\n测试 Gauss-Newton VP...")
    gv = gauss_newton_vector_product(model, x, y, v)
    print(f"  输入向量范数: {v.norm().item():.4f}")
    print(f"  输出向量范数: {gv.norm().item():.4f}")

    # 测试 HVP
    print("\n测试 Hessian VP...")
    hv = hessian_vector_product(model, x, y, v)
    print(f"  输入向量范数: {v.norm().item():.4f}")
    print(f"  输出向量范数: {hv.norm().item():.4f}")

    # 测试对称性：<v, Hv> 应该等于 <Hv, v>
    print("\n测试对称性...")
    v2 = torch.randn(n_params)
    hv1 = hessian_vector_product(model, x, y, v)
    hv2 = hessian_vector_product(model, x, y, v2)

    sym1 = torch.dot(v, hv2).item()
    sym2 = torch.dot(v2, hv1).item()
    rel_diff = abs(sym1 - sym2) / max(abs(sym1), abs(sym2), 1e-10)

    print(f"  <v1, H·v2> = {sym1:.6f}")
    print(f"  <v2, H·v1> = {sym2:.6f}")
    print(f"  相对差异: {rel_diff:.2e}")

    if rel_diff < 1e-4:
        print("\n✅ 测试通过！HVP 实现正确")
    else:
        print("\n⚠️  警告：对称性误差较大")

    print("=" * 80)


if __name__ == "__main__":
    test_hvp_on_toy_model()
