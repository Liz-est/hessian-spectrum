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
# Muon 数学（NS5 正交化 + C5 预条件器递推），训练与谱分析共用（不建 muon.py）。
# ============================================================================
_NS_ABC = (3.4445, -4.7750, 2.0315)     # Moonlight/Muon quintic 系数


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


def muon_precond_C5(G, steps=5, eps_ns=1e-7):
    """按 markdown 递推求 C₅ = P⁻¹（NS5-忠实版，非理想极分解）。

    设 X_k = G·C_k（tall，右乘）或 X_k = C_k·G（wide，左乘），C_k 为 S 的多项式
    （对称、与 S 交换），则 NS5 迭代 X_{k+1}=a·X_k+b·(X_kX_kᵀ)X_k+c·(X_kX_kᵀ)²X_k
    恰好塌缩为标量递推：
        C_{k+1} = a·C_k + b·S·C_k³ + c·S²·C_k⁵,   C_0 = I/α,  α = ‖G‖_F + eps_ns
    tall(m≥n): S = GᵀG (n×n)，side='right'；wide: S = GGᵀ (m×m)，side='left'。
    返回 (C5, side)。C5 即优化器右/左乘的预条件器（含 NS5 近似误差）。fp32。

    ⚠ G 为**单层 2D 矩阵**（分析端逐层调用）；α 为该层 Frobenius 范数，与 batched
    NS5 的逐层归一化一致。
    """
    a, b, c = _NS_ABC
    G = G.float()
    m, n = G.shape[-2], G.shape[-1]
    alpha = G.norm() + eps_ns
    if m >= n:
        S = G.transpose(-2, -1) @ G      # (n,n) = GᵀG
        side, d = "right", n
    else:
        S = G @ G.transpose(-2, -1)      # (m,m) = GGᵀ
        side, d = "left", m
    C = torch.eye(d, dtype=G.dtype, device=G.device) / alpha    # C_0 = I/α
    S2 = S @ S
    for _ in range(steps):
        C2 = C @ C
        C3 = C2 @ C
        C5 = C3 @ C2
        C = a * C + b * (S @ C3) + c * (S2 @ C5)
    return C, side


def muon_Phalf(G, pre, post, steps=5):
    """谱分析用的对称半预条件器：Phalf = √(pre·post) · C₅^{1/2}（小边 d×d）。

    G = 存下的（已 pre-scaled）动量 buffer 的逐层 2D 矩阵。HVP 两侧各施加一次
    Phalf → 全 metric = pre·post·C₅ = post·C₅(G₀)（与 Adam 的 pre·post/√ν̂ 同构）。
    返回 (Phalf, side)，side∈{'right'(tall),'left'(wide)}。
    """
    C5, side = muon_precond_C5(G, steps)
    C5 = 0.5 * (C5 + C5.transpose(-2, -1))                      # 对称化
    evals, evecs = torch.linalg.eigh(C5)
    evals = evals.clamp_min(0.0)
    C5_half = (evecs * evals.sqrt()) @ evecs.transpose(-2, -1)   # C₅^{1/2}
    return math.sqrt(pre * post) * C5_half, side


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


class CompletePMuon:
    """
    在 (name, param, pre, post) 四元组列表上运行的 CompleteP-Muon（Moonlight NS5）。

    结构对齐 CompletePAdam（同吃 lr_groups、同 CosineSchedule、step() 返回 base_lr），
    差别在更新规则：
      - 梯度 pre 缩放：u = pre·g（与 Adam 对称；pre 会在 msign 里精确相消，但保留于
        buffer 使动量与 Adam 同口径、并让存下的 buffer 直接可用于分析端 √(pre·post)）。
      - 动量 buffer：buf ← momentum·buf + u（Moonlight 约定）。
      - 更新方向：g_eff = u + momentum·buf（nesterov）或 buf；逐层 msign = NS5(g_eff)。
      - 更新：Δθ = −base_lr · post · scale · msign，scale=1（默认）或 0.2·√max(m,n)
        （rms_match=True，Moonlight RMS 匹配口径）。
    stacked 参数按 muon_reshape 拆成 (L,m,n) 逐层正交化。无 weight decay。
    """
    def __init__(self, lr_groups, lr, total_steps,
                 momentum=0.95, nesterov=True, ns_steps=5, rms_match=False,
                 schedule_kwargs=None):
        self.groups = lr_groups
        self.names = [name for name, _, _, _ in lr_groups]
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.rms_match = rms_match
        self.schedule = CosineSchedule(total_steps, **(schedule_kwargs or {}))
        self.count = 0
        self.buf = [torch.zeros_like(p, dtype=torch.float32) for _, p, _, _ in self.groups]

    @torch.no_grad()
    def step(self):
        self.count += 1
        base_lr = self.schedule(self.count) * self.lr
        for i, (name, p, pre, post) in enumerate(self.groups):
            if p.grad is None:
                continue
            u = p.grad.detach().float() * pre                   # pre 缩放（对称 Adam）
            self.buf[i].mul_(self.momentum).add_(u)             # buf ← mom·buf + u
            g_eff = u.add(self.buf[i], alpha=self.momentum) if self.nesterov else self.buf[i]
            # 逐层 2D reshape → NS5 正交化（batched 对 (L,m,n) 的最后两维）
            g2d = muon_reshape(name, g_eff)                     # (L,m,n)
            msign = zeropower_via_newtonschulz5(g2d, self.ns_steps)
            if self.rms_match:
                m2, n2 = g2d.shape[-2], g2d.shape[-1]
                scale = 0.2 * math.sqrt(max(m2, n2))
            else:
                scale = 1.0
            upd = msign.reshape(p.shape).to(p.dtype)
            p.add_(upd, alpha=-base_lr * post * scale)
        return base_lr

    def zero_grad(self, set_to_none=True):
        for _, p, _, _ in self.groups:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def get_buf(self):
        """返回 {name: buf}（已 pre-scaled 的动量 buffer），供 EMA 与谱分析。"""
        return {name: self.buf[i].clone()
                for i, (name, _, _, _) in enumerate(self.groups)}

    def state_for_ckpt(self):
        """存 checkpoint：动量 buffer（逐组）+ 标量 + 每组 pre/post，供谱分析重建
        Muon 预条件器 √(pre·post)·C₅^{1/2}（见 spectrum_ddp.build_muon_precond）。"""
        return dict(
            name="muon",
            count=self.count,
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
            ns_steps=self.ns_steps,
            rms_match=self.rms_match,
            buf={name: self.buf[i].cpu() for i, (name, _, _, _) in enumerate(self.groups)},
            pre={name: pre for name, _, pre, _ in self.groups},
            post={name: post for name, _, _, post in self.groups},
            base_lr_final=self.schedule(self.count) * self.lr,
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


if __name__ == "__main__":
    import model as M
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(0)

    # ---- 1. NS5 & C5 一致性自检 ----
    print("=" * 70)
    print("1. NS5 正交化 & C₅ 递推自检")
    for (m, n) in [(1024, 1024), (2048, 512), (512, 2048)]:
        G = torch.randn(m, n)
        msign = zeropower_via_newtonschulz5(G, 5)          # ≈ UVᵀ
        # 奇异值应 ≈ 1（正交化）
        sv = torch.linalg.svdvals(msign.float())
        # C₅ 一致性：X₅ = G·C₅（tall）或 C₅·G（wide）应 ≈ NS5(G)
        C5, side = muon_precond_C5(G, 5)
        X5 = G @ C5 if side == "right" else C5 @ G
        rel = (X5 - msign.float()).norm() / (msign.float().norm() + 1e-9)
        # Phalf 对称性 & Phalf·C₅⁻¹·Phalf ≈ pre·post·I
        Phalf, side2 = muon_Phalf(G, pre=2.0, post=3.0, steps=5)
        C5inv = torch.linalg.inv(C5)
        chk = Phalf @ C5inv @ Phalf                        # 应 ≈ pre·post·I = 6·I
        eye_err = (chk - 6.0 * torch.eye(chk.size(0))).norm() / chk.norm()
        print(f"  G({m}x{n}) side={side}: σ(msign)∈[{sv.min():.3f},{sv.max():.3f}]  "
              f"‖X₅−NS5‖/‖NS5‖={rel:.2e}  Phalz对称={torch.allclose(Phalf,Phalf.T,atol=1e-4)}  "
              f"Phalf·C₅⁻¹·Phalf≈6I err={eye_err:.2e}")

    # ---- 2. CompletePMuon 数值自检 ----
    print("=" * 70)
    print("2. CompletePMuon 训练 3 步自检")
    cfg = M.TransformerConfig(D=64, L=2, M=128, H=2, K=32, V=128, seq_len=16)
    net = M.Transformer(cfg)
    opt = CompletePMuon(net.lr_groups(), lr=1e-2, total_steps=100)
    x = torch.randint(0, cfg.V, (2, cfg.seq_len))
    for step in range(3):
        opt.zero_grad()
        with sdpa_kernel([SDPBackend.MATH]):
            _, loss = net(x, x)
        loss.backward()
        # 记录 head 更新前后差，验证 ‖Δθ‖ 与 base_lr·post 量级
        head_before = net.head.detach().clone()
        blr = opt.step()
        dnorm = (net.head.detach() - head_before).norm().item()
        print(f"  step {step+1}: loss={loss.item():.4f} base_lr={blr:.3e} ‖Δhead‖={dnorm:.3e}")

    # pre 相消验证：msign(pre·G) == msign(G)
    G = torch.randn(256, 256)
    m1 = zeropower_via_newtonschulz5(G, 5)
    m2 = zeropower_via_newtonschulz5(7.3 * G, 5)
    print(f"  pre 相消: ‖msign(G)−msign(7.3·G)‖/‖msign(G)‖="
          f"{(m1-m2).norm()/m1.norm():.2e}  (应≈0)")

    # ---- 3. CompletePAdam 回归自检（不破坏原有）----
    print("=" * 70)
    print("3. CompletePAdam 回归")
    net2 = M.Transformer(cfg)
    opt2 = CompletePAdam(net2.lr_groups(), lr=1e-3, total_steps=100)
    for step in range(3):
        opt2.zero_grad()
        with sdpa_kernel([SDPBackend.MATH]):
            _, loss = net2(x, x)
        loss.backward()
        blr, b2 = opt2.step()
        print(f"  step {step+1}: loss={loss.item():.4f} base_lr={blr:.3e} beta2={b2:.5f}")
    print(f"  state name={opt2.state_for_ckpt()['name']}  "
          f"muon state name={opt.state_for_ckpt()['name']}")
