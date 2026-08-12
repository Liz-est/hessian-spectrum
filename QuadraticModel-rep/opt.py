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
    # 数值对齐自检：单参数与手算 Adam 对比
    import model as M
    cfg = M.TransformerConfig(D=64, L=2, M=128, H=2, K=32, V=128, seq_len=16)
    net = M.Transformer(cfg)
    opt = CompletePAdam(net.lr_groups(), lr=1e-3, total_steps=100)
    x = torch.randint(0, cfg.V, (2, cfg.seq_len))
    from torch.nn.attention import sdpa_kernel, SDPBackend
    for step in range(3):
        opt.zero_grad()
        with sdpa_kernel([SDPBackend.MATH]):
            _, loss = net(x, x)
        loss.backward()
        blr, b2 = opt.step()
        print(f"step {step+1}: loss={loss.item():.4f} base_lr={blr:.3e} beta2={b2:.5f}")
    nh = opt.get_nu_hat()
    print("nu_hat keys:", list(nh.keys()))
    print("head nu_hat mean:", nh["head"].mean().item())
