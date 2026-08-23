"""
CompleteP Adam 的 PyTorch 移植，严格对齐 JAX 参考 QuadraticModel/opt.py。

要点（逐行对照 JAX）：
  - 每参数组有 (pre, post) 学习率乘子（来自 model.lr_groups()，含 CompleteP + BlockScales）
  - 梯度先做 pre 缩放：u = g * lr.pre
  - 一阶矩 mu、二阶矩 nu 都作用在 pre-缩放后的 u 上
  - 动态 β2：m = max(20, b2_pct*count)，β2_t = 1 - 1/m（b2_pct=0.01）
  - log 空间偏差校正：mu_hat = mu/(1 - b1^count)，nu_hat = nu/(1 - Πβ2_t)
  - 更新：raw = base_lr * mu_hat/(√nu_hat + eps)，Δθ = -lr.post * raw
  - base_lr = cosine_schedule(count) * lr（warmup_pct=0.2, init 0.1, peak 1.0, end 0.0）
  - 无 weight decay

EMA：对 pct 列表里每个 ρ 维护一份参数 EMA（ρ_t 见 EMA.update）。
预条件器复原：spectrum 分析用 get_nu_hat()（bias_correct(nu, log_prod_b2)）+ 存下的
  base_lr / lr.pre / lr.post / eps 精确重建 Adam 度量。
"""
from __future__ import annotations
import math
import torch


# ============================================================================
# Muon 数学（NS5 正交化 + C5 预条件器递推），训练与谱分析共用。
# ============================================================================
_NS_ABC = (3.4445, -4.7750, 2.0315)     # Moonlight/Muon quintic 系数（训练用）


# Polar Express 系数（arXiv 2505.16932, safety_factor=1.05 缩放）—— Gram NS 递推用。
# 复刻自 Dao-AILab/gram-newton-schulz/gram_newton_schulz/coefficients.py。
_POLAR_EXPRESS_RAW = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
_SF = 1.05
POLAR_EXPRESS_COEFFICIENTS = [
    (a / _SF, b / _SF**3, c / _SF**5)
    for (a, b, c) in _POLAR_EXPRESS_RAW
]


def gram_newton_schulz_C5(G, steps=5, eps_ns=1e-7, reset_iterations=(2,)):
    """Gram Newton-Schulz 迭代求 C5 = (G^T G)^{-1/2}（tall）或 (G G^T)^{-1/2}（wide）。

    复刻自 Dao-AILab/gram-newton-schulz（POLAR_EXPRESS 系数），保留 reset 机制：
    每 reset_iterations 步从更新后的 X 重算 Gram 矩阵 R，防止近奇异矩阵上的数值爆炸。

    数学：先归一化 X = G/‖G‖，Gram 迭代得 Q = (X^T X)^{-1/2} = ‖G‖·(G^T G)^{-1/2}，
    除以 ‖G‖ 恢复：C5 = Q / ‖G‖。

    G 为单层 2D 矩阵（fp32）。返回 (C5, side)，side ∈ {'right'(tall), 'left'(wide)}。
    """
    tall = G.size(-2) >= G.size(-1)   # m≥n=tall（与 eigh 版 side 判定对齐）
    norm_G = G.float().norm()
    X = G.float() / (norm_G + eps_ns)   # 归一化

    if tall:
        R = X.mT @ X              # (n,n)
    else:
        R = X @ X.mT              # (m,m)
    R = 0.5 * (R + R.mT)

    d = R.size(-1)
    I = torch.eye(d, dtype=R.dtype, device=R.device)
    Q = None
    reset_set = set(reset_iterations)
    coeffs = POLAR_EXPRESS_COEFFICIENTS[:steps]

    for i, (a, b, c) in enumerate(coeffs):
        if i in reset_set and i != 0:
            if tall:
                X = X @ Q
                R = X.mT @ X
            else:
                X = Q @ X
                R = X @ X.mT
            R = 0.5 * (R + R.mT)
            Q = None

        Z = b * R + c * (R @ R)
        Z = 0.5 * (Z + Z.mT)

        if i == 0 or i in reset_set:
            Q = Z + a * I
        else:
            # 参考代码: sym_baddbmm(Q, Z, C=Q, beta=a) = Q@Z + a*Q（乘积累积，
            # 每步都作用在 R 的特征值上——正是收敛到 R^{-1/2} 的关键）
            Q = Q @ Z + a * Q

        if i < len(coeffs) - 1 and (i + 1) not in reset_set:
            RZ = R @ Z + a * R
            R_new = Z @ RZ + a * RZ
            R = 0.5 * (R_new + R_new.mT)

    # 除以归一化因子：Q = ‖G‖·C5 → C5 = Q/‖G‖
    C5 = Q / norm_G
    return C5, ("right" if tall else "left")


def muon_ns5_C5(G, steps=5, eps_ns=1e-7):
    """训练端 NS5 的**隐含预条件器** C₅，使 NS5(G) = G·C₅（tall）或 C₅·G（wide）。

    定义 NS5 的每步为
        Xₖ₊₁ = a·Xₖ + (b·Aₖ + c·Aₖ²)·Xₖ,  Aₖ = XₖXₖᵀ,
    等价于右乘一个 Xₖ 的 Gram 的多项式：Xₖ₊₁ = Xₖ·Pₖ，Pₖ = a·I + b·Rₖ + c·Rₖ²，
    Rₖ = XₖᵀXₖ（tall）。5 步累积 Q = ∏ₖ Pₖ，则 NS5(G) = X·Q = (G/‖G‖)·Q，
    故 C₅ = Q/‖G‖。同理 wide 用 Rₖ = XₖXₖᵀ 左乘。

    **数值稳定的关键**：多项式在**归一化后 Gram 矩阵**（谱 ⊂ [0,1]）上累积，而非
    在每个奇异值上独立跑标量高次递推——数学上与标量递推恒等（同一 Moonlight NS5），
    但不会因 σ≪1 时 (1/σ²)² 溢出。用 Moonlight 系数 _NS_ABC、**无 reset**（训练端
    NS5 就没有 reset）。C₅ 严格对称且在实测真实 buffer 上正定。

    G 为单层 2D 矩阵（fp32）。返回 (C5, side)，side∈{'right'(tall),'left'(wide)}。
    """
    a, b, c = _NS_ABC
    tall = G.size(-2) >= G.size(-1)
    Gd = G.double()
    norm_G = Gd.norm()
    X = Gd / (norm_G + eps_ns)
    R = X.mT @ X if tall else X @ X.mT       # 小边归一化 Gram，谱 ⊂ [0,1]
    R = 0.5 * (R + R.mT)
    d = R.size(-1)
    I = torch.eye(d, dtype=R.dtype, device=R.device)
    Q = I.clone()
    for _ in range(steps):
        P = a * I + b * R + c * (R @ R)      # 本步多项式（R 的多项式 → 对称）
        Q = Q @ P                            # 乘积累积（P 互相对易 → Q 仍对称）
        R = P @ R @ P                        # R_{k+1} = P Rₖ P（= Xₖ₊₁ 的 Gram）
        R = 0.5 * (R + R.mT)
    C5 = Q / norm_G                          # 还原归一化：NS5(G)=X·Q → C₅=Q/‖G‖
    C5 = 0.5 * (C5 + C5.mT)
    return C5.to(G.dtype), ("right" if tall else "left")


def zeropower_via_newtonschulz5(G, steps=5):
    """G 的 5 步 Newton-Schulz 谱正交化 ≈ msign(G)=UVᵀ（G=USVᵀ）。
    与 toy_models/config/build.py 逐字对齐：bf16、tall 转置、Frobenius 归一化。
    支持 batched（对最后两维操作，首维视作层 L）。"""
    assert G.ndim >= 2
    a, b, c = _NS_ABC
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def muon_Phalf(G, pre, post, steps=5):
    """谱分析用的对称半预条件器：Phalf = √(pre·post) · C₅^{1/2}（小边 d×d）。

    G = 存下的（已 pre-scaled）动量 buffer 的逐层 2D 矩阵。HVP 两侧各施加一次
    Phalf → 全 metric = pre·post·C₅ = post·C₅(G₀)（与 Adam 的 pre·post/√ν̂ 同构）。
    返回 (Phalf, side)，side∈{'right'(tall),'left'(wide)}。

    C₅ 用 muon_ns5_C5：训练端 NS5 隐含的预条件器（忠实复现，非 eigh 的精确
    (GᵀG)^{-1/2}——后者带训练里不存在的 rel_floor 正则，且在真实 buffer 上差 2–8×）。
    """
    C5, side = muon_ns5_C5(G, steps)
    C5 = 0.5 * (C5 + C5.transpose(-2, -1))                      # 对称化
    C5_half = _sym_psd_sqrt(C5)                                  # 抗病态对称平方根
    return math.sqrt(pre * post) * C5_half, side


def _sym_psd_sqrt(A):
    """对称半正定矩阵的对称平方根 A^{1/2}，抗病态。

    ⚠ C₅ 逼近正交投影（特征值聚在 {0,1}），高度简并 → 分解易不收敛。且 GPU
    LAPACK（cusolver）对简并矩阵的 eigh/svd 都常崩（error 990/1010/1024…，
    lr0.32 ckpt 全 rank 挂在此）。两条应对：
      (a) **搬到 CPU 算分解**：CPU LAPACK 对病态矩阵远比 cusolver 鲁棒（小 d×d，开销可忽略）；
      (b) 分级兜底：eigh(fp64) → 加递增 jitter 破简并 → SVD。
    结果搬回原 device/dtype。"""
    dt, dev = A.dtype, A.device
    Ad = A.detach().double().cpu()                              # CPU + fp64
    Ad = 0.5 * (Ad + Ad.transpose(-2, -1))
    d = Ad.shape[-1]
    eye = torch.eye(d, dtype=Ad.dtype)
    H = None
    for jit in (0.0, 1e-9, 1e-7, 1e-5, 1e-3):
        try:
            evals, evecs = torch.linalg.eigh(Ad + jit * eye)
            evals = (evals - jit).clamp_min(0.0)               # 抵消 jitter 偏移
            H = (evecs * evals.sqrt()) @ evecs.transpose(-2, -1)
            break
        except torch._C._LinAlgError:
            continue
    if H is None:
        # SVD 兜底：A 对称半正定 → A = U diag(S) Uᵀ（U=V），A^{1/2}=U diag(√S) Uᵀ
        U, S, _ = torch.linalg.svd(Ad)
        H = (U * S.sqrt()) @ U.transpose(-2, -1)
    H = 0.5 * (H + H.transpose(-2, -1))
    return H.to(device=dev, dtype=dt)


# 每个权重参数的「逐层 2D 矩阵」reshape → (L, m, n)。stacked 参数首维为层 L；
# embd/head 无 L（视作 L=1）。Muon 的 msign 与预条件器都按此约定，二者必须一致。
def muon_reshape(name, t):
    if name in ("attn_q", "attn_k", "attn_v"):    # (L,D,H,K) -> (L, D, H*K)
        L, D, H, K = t.shape
        return t.reshape(L, D, H * K)
    if name == "attn_head":                        # (L,H,K,D) -> (L, H*K, D)
        L, H, K, D = t.shape
        return t.reshape(L, H * K, D)
    if t.ndim == 3:                                # mlp_up (L,D,M) / mlp_head (L,M,D)
        return t
    if t.ndim == 2:                                # embd (V,D) / head (D,V)
        return t.unsqueeze(0)
    raise ValueError(f"muon_reshape: 未知参数 {name} shape {tuple(t.shape)}")


class CosineSchedule:
    def __init__(self, steps, warmup_pct=0.2, init_value=0.1, peak_value=1.0, end_value=0.0):
        self.steps = steps
        self.warmup_pct = warmup_pct
        self.init_value = init_value
        self.peak_value = peak_value
        self.end_value = end_value

    def __call__(self, t):
        s = t / self.steps
        wp = self.warmup_pct
        warmup = self.init_value + (self.peak_value - self.init_value) * (s / wp)
        t_cos = min((s - wp) / (1 - wp), 1.0)
        cos = 0.5 * (1 + math.cos(math.pi * t_cos))
        cosine = self.end_value + (self.peak_value - self.end_value) * cos
        return warmup if s < wp else cosine


class CompletePAdam:
    """
    在 (param, pre, post) 三元组列表上运行的自定义 Adam。
    与 torch.optim.Optimizer 不同，这里手动管理状态以保证与 JAX 逐位对齐。
    """
    def __init__(self, lr_groups, lr, total_steps,
                 b1=0.9, b2_pct=0.01, eps=1e-8,
                 schedule_kwargs=None):
        # lr_groups: list of (name, param, pre, post)
        self.groups = lr_groups
        self.lr = lr
        self.b1 = b1
        self.b2_pct = b2_pct
        self.eps = eps
        self.schedule = CosineSchedule(total_steps, **(schedule_kwargs or {}))
        self.count = 0
        self.log_prod_b2 = 0.0
        self.mu = [torch.zeros_like(p, dtype=torch.float32) for _, p, _, _ in self.groups]
        self.nu = [torch.zeros_like(p, dtype=torch.float32) for _, p, _, _ in self.groups]

    @torch.no_grad()
    def step(self):
        self.count += 1
        count = self.count
        base_lr = self.schedule(count) * self.lr

        m = max(20.0, self.b2_pct * count)
        beta2_t = 1.0 - 1.0 / m
        self.log_prod_b2 += math.log(beta2_t)

        denom_b1 = -math.expm1(count * math.log(self.b1))       # 1 - b1^count
        denom_b1 = max(denom_b1, 1e-16)
        denom_b2 = -math.expm1(self.log_prod_b2)                # 1 - Πβ2
        denom_b2 = max(denom_b2, 1e-16)

        for i, (name, p, pre, post) in enumerate(self.groups):
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            u = g * pre                                         # pre 缩放
            self.mu[i].lerp_(u, 1 - self.b1)                    # mu += (u-mu)*(1-b1)
            self.nu[i].lerp_(u * u, 1 - beta2_t)
            mu_hat = self.mu[i] / denom_b1
            nu_hat = self.nu[i] / denom_b2
            raw = base_lr * mu_hat / (nu_hat.sqrt() + self.eps)
            p.add_((-post * raw).to(p.dtype))
        return base_lr, beta2_t

    def zero_grad(self, set_to_none=True):
        for _, p, _, _ in self.groups:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def get_nu_hat(self):
        """bias-corrected ν̂，用于 Adam 预条件器（与 JAX get_nu_hat 一致）。"""
        denom_b2 = max(-math.expm1(self.log_prod_b2), 1e-16)
        return {name: (self.nu[i] / denom_b2).clone()
                for i, (name, _, _, _) in enumerate(self.groups)}

    def state_named(self):
        """[(name, ν)]：优化器状态 EMA 用（Adam 追踪二阶矩 ν）。"""
        return [(name, self.nu[i]) for i, (name, _, _, _) in enumerate(self.groups)]

    def state_for_ckpt(self):
        """存 checkpoint：ν 状态 + 标量 + 每组 pre/post，供谱分析重建预条件器。"""
        return dict(
            name="adam",
            count=self.count,
            log_prod_b2=self.log_prod_b2,
            lr=self.lr,
            eps=self.eps,
            nu={name: self.nu[i].cpu() for i, (name, _, _, _) in enumerate(self.groups)},
            mu={name: self.mu[i].cpu() for i, (name, _, _, _) in enumerate(self.groups)},
            pre={name: pre for name, _, pre, _ in self.groups},
            post={name: post for name, _, _, post in self.groups},
            base_lr_final=self.schedule(self.count) * self.lr,
        )


class Muon:
    """本仓库 "muon" 优化器：embd/head 走 CompletePAdam（沿用 Adam+CompleteP），
    其余 6 层（attn_q/k/v/attn_head、mlp_up/mlp_head）走 Moonlight NS5 谱正交化。

    动机：纯 Muon 在 embedding/lm_head（秩结构与 softmax 类不平衡）上白化几乎无效
    且 rms_match 口径难调；这两层沿用 Adam+CompleteP 的成熟口径，隐藏层矩阵仍由
    Muon 谱正交化。embd/head 与隐藏层各自独立的 base_lr（adam_lr vs lr）与
    CosineSchedule（同 total_steps/schedule_kwargs）。

    隐藏层更新规则（逐层，与 Adam 对称的 pre 缩放）：
      - 梯度 pre 缩放：u = pre·g（pre 在 msign 里精确相消，但保留于 buffer 使动量与
        Adam 同口径、并让存下的 buffer 直接可用于分析端 √(pre·post)）。
      - 动量 buffer：buf ← momentum·buf + u（Moonlight 约定）。
      - 更新方向：g_eff = u + momentum·buf（nesterov）或 buf；逐层 msign = NS5(g_eff)。
      - 更新：Δθ = −base_lr · post · scale · msign。
    两套「每层尺度」互斥旋钮（completep 与 rms_match）：
      - completep=True：用 CompleteP 的 (pre,post) 乘子（u=pre·g，Δθ 带 post）。
      - completep=False：忽略 (pre,post)（置 1），改用 Moonlight RMS 匹配
        scale=0.2·√max(m,n)（要求 rms_match=True），lr 轴与 Adam 对齐（RMS≈0.2）。
    rms_match 在 completep=True 时仍可叠加，但常规二选一。stacked 参数按 muon_reshape
    拆成 (L,m,n) 逐层正交化。无 weight decay。

    state_for_ckpt 记 name="muon" + adam_names/muon_names + adam/muon 子状态，
    谱分析据此对每层分别用 Adam 对角 / Muon dense-op 预条件器（precond dict 混装，
    hvp 已按 name 分派；见 spectrum_ddp.load_checkpoint）。
    """
    ADAM_LAYERS = ("embd", "head")

    def __init__(self, lr_groups, lr, adam_lr, total_steps,
                 b1=0.9, b2_pct=0.01, eps=1e-8,
                 momentum=0.95, nesterov=True, ns_steps=5,
                 muon_completep=True, muon_rms_match=False,
                 schedule_kwargs=None):
        # embd/head → 内部 CompletePAdam；其余 6 层 → Muon NS5（本类内联）
        adam_groups = [g for g in lr_groups if g[0] in self.ADAM_LAYERS]
        self.muon_groups = [g for g in lr_groups if g[0] not in self.ADAM_LAYERS]
        self.adam = CompletePAdam(adam_groups, lr=adam_lr, total_steps=total_steps,
                                  b1=b1, b2_pct=b2_pct, eps=eps,
                                  schedule_kwargs=schedule_kwargs)
        self.adam_names = [g[0] for g in adam_groups]
        self.muon_names = [g[0] for g in self.muon_groups]
        # 隐藏层 Muon 状态
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.completep = muon_completep
        self.rms_match = muon_rms_match
        self.schedule = CosineSchedule(total_steps, **(schedule_kwargs or {}))
        self.count = 0
        self.buf = [torch.zeros_like(p, dtype=torch.float32) for _, p, _, _ in self.muon_groups]

    @torch.no_grad()
    def step(self):
        blr_a, _ = self.adam.step()                             # embd/head 走 Adam
        self.count += 1
        base_lr = self.schedule(self.count) * self.lr
        for i, (name, p, pre, post) in enumerate(self.muon_groups):
            if p.grad is None:
                continue
            # completep 关时忽略 (pre,post) 乘子（置 1），纯靠 rms_match 定每层尺度
            pre_eff = pre if self.completep else 1.0
            post_eff = post if self.completep else 1.0
            u = p.grad.detach().float() * pre_eff               # pre 缩放（对称 Adam）
            self.buf[i].mul_(self.momentum).add_(u)             # buf ← mom·buf + u
            g_eff = u.add(self.buf[i], alpha=self.momentum) if self.nesterov else self.buf[i]
            # 逐层 2D reshape → NS5 正交化（batched 对 (L,m,n) 的最后两维）
            g2d = muon_reshape(name, g_eff)                     # (L,m,n)
            msign = zeropower_via_newtonschulz5(g2d, self.ns_steps)
            scale = 1.0
            if self.rms_match:
                m2, n2 = g2d.shape[-2], g2d.shape[-1]
                scale *= 0.2 * math.sqrt(max(m2, n2))
            upd = msign.reshape(p.shape).to(p.dtype)
            p.add_(upd, alpha=-base_lr * post_eff * scale)
        return blr_a, base_lr

    def zero_grad(self, set_to_none=True):
        self.adam.zero_grad(set_to_none)
        for _, p, _, _ in self.muon_groups:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def state_named(self):
        """[(name, state)]：embd/head 给 ν、其余 6 层给 buf；供优化器状态 EMA（混装）。"""
        muon_state = [(name, self.buf[i])
                      for i, (name, _, _, _) in enumerate(self.muon_groups)]
        return self.adam.state_named() + muon_state

    def _muon_state_for_ckpt(self):
        """隐藏层 Muon 子状态：动量 buffer + 标量 + 每层 pre/post，供谱分析重建
        dense-op 预条件器 √(pre·post)·C₅^{1/2}（见 spectrum_ddp._muon_precond_from_buf）。"""
        return dict(
            count=self.count,
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
            ns_steps=self.ns_steps,
            rms_match=self.rms_match,
            completep=self.completep,
            buf={name: self.buf[i].cpu() for i, (name, _, _, _) in enumerate(self.muon_groups)},
            pre={name: pre for name, _, pre, _ in self.muon_groups},
            post={name: post for name, _, _, post in self.muon_groups},
            base_lr_final=self.schedule(self.count) * self.lr,
        )

    def state_for_ckpt(self):
        return dict(
            name="muon",
            adam_names=list(self.adam_names),
            muon_names=list(self.muon_names),
            adam=self.adam.state_for_ckpt(),
            muon=self._muon_state_for_ckpt(),
        )


class EMA:
    """对多个 ρ 维护参数 EMA，对齐 JAX opt.EMA。"""
    def __init__(self, params_named, pct=(0.04,), dtype=torch.float32):
        self.pct = list(pct)
        self.dtype = dtype
        self.count = 0
        self.ema = {p: {name: v.detach().to(dtype).clone()
                        for name, v in params_named}
                    for p in self.pct}

    @torch.no_grad()
    def update(self, params_named):
        pn = {name: v for name, v in params_named}
        for p in self.pct:
            h = max(1.0, self.count * p / (1 - p))
            s = 1.0 / h
            for name, cur in self.ema[p].items():
                cur.lerp_(pn[name].detach().to(self.dtype), s)
        self.count += 1

    def state_for_ckpt(self):
        return {str(p): {name: v.cpu() for name, v in self.ema[p].items()}
                for p in self.pct}