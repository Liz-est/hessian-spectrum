"""
Builders that turn config blocks into live torch objects.

Keeping these here (rather than inline in the trainer) means "which optimizer /
which LR schedule" is fully driven by OptimConfig / LRConfig -- adding a new
optimizer is one branch here, not an edit to the training loop.
"""

import math

import torch


# ---------------------------------------------------------------------------
# Muon (MomentUm Orthogonalized by Newton-Schulz), Moonlight version
# (Moonshot AI, arXiv 2502.16982). Per-matrix update:
#
#     W_t = W_{t-1} - lr * ( 0.2 * sqrt(max(d_out, d_in)) * msign(M_t)
#                            + weight_decay * W_{t-1} )
#
# where M_t is the (nesterov) momentum buffer and msign is the Newton-Schulz
# orthogonalization. The 0.2*sqrt(max(A,B)) factor matches the update RMS to
# AdamW's (~0.2), so Muon shares AdamW's lr / weight_decay -- no separate
# tuning. Muon only makes sense for the hidden >=2D weight matrices;
# embeddings, lm_head and all 1D params (biases, LayerNorms) go to an internal
# AdamW param group inside the same optimizer object, so the trainer's
# per-step `for g in optimizer.param_groups: g["lr"] = get_lr(it)` loop keeps
# working unchanged. Each group applies its own `lr_scale` on top of the
# scheduled lr (muon_aux_lr_scale, default 1.0 since the RMS-matched scaling
# means both groups want the same lr).
# ---------------------------------------------------------------------------
def _zeropower_via_newtonschulz5(G, steps):
    """Approximate orthogonalization of G via quintic Newton-Schulz iteration.

    Returns ~ U V^T for G = U S V^T (msign). Coefficients as in Muon/Moonlight;
    run in bfloat16 for speed (a loose approximation is fine here).
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
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


class Muon(torch.optim.Optimizer):
    """Moonlight-style Muon for hidden matrices + built-in AdamW for the rest."""

    def __init__(self, muon_params, aux_params, lr=0.0, weight_decay=0.0,
                 momentum=0.95, nesterov=True, ns_steps=5,
                 betas=(0.9, 0.95), eps=1e-8, aux_lr_scale=1.0):
        param_groups = [
            dict(params=list(muon_params), use_muon=True, lr_scale=1.0,
                 momentum=momentum, nesterov=nesterov, ns_steps=ns_steps),
            dict(params=list(aux_params), use_muon=False, lr_scale=aux_lr_scale,
                 betas=betas, eps=eps),
        ]
        super().__init__(param_groups, dict(lr=lr, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"] * group["lr_scale"]
            wd = group["weight_decay"]
            if group["use_muon"]:
                mom = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)
                    # Moonlight momentum: M_t = mom * M_{t-1} + g_t
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(p.grad)
                    upd = p.grad.add(buf, alpha=mom) if group["nesterov"] else buf
                    upd = _zeropower_via_newtonschulz5(upd, group["ns_steps"])
                    # Moonlight RMS-matched scale: 0.2 * sqrt(max(d_out, d_in))
                    scale = 0.2 * math.sqrt(max(p.size(-2), p.size(-1)))
                    # W -= lr * (scale * msign(M) + wd * W), i.e. decoupled wd
                    p.mul_(1 - lr * wd)
                    p.add_(upd, alpha=-lr * scale)
            else:
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p.grad)
                        state["exp_avg_sq"] = torch.zeros_like(p.grad)
                    state["step"] += 1
                    t = state["step"]
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.lerp_(p.grad, 1 - beta1)
                    v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)
                    p.mul_(1 - lr * wd)
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


def build_optimizer(params, cfg):
    """Construct a torch optimizer from an OptimConfig.

    `params` is anything torch accepts (model.parameters()) -- or the model
    itself, which is REQUIRED for "muon": Muon needs parameter names/shapes to
    route hidden matrices to the Muon group and embeddings / lm_head / 1D
    params to its internal AdamW group. Each branch reads only the hyper-params
    that optimizer actually uses; the rest of OptimConfig is ignored, so e.g.
    `betas` is harmless for plain SGD.
    """
    name = cfg.name.lower()
    lr = 0.0  # real LR is set per-step by the scheduler; start at 0 for warmup
    module = None
    if isinstance(params, torch.nn.Module):
        module, params = params, params.parameters()
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=cfg.momentum, nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=lr, betas=tuple(cfg.betas), eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            params, lr=lr, betas=tuple(cfg.betas), eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    if name == "muon":
        if module is None:
            raise ValueError(
                "optimizer 'muon' needs parameter names: call "
                "build_optimizer(model, cfg) with the model, not .parameters()"
            )
        muon_params, aux_params = [], []
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            # hidden matrices -> Muon; embeddings / lm_head / 1D -> aux AdamW
            if p.ndim >= 2 and "tok_emb" not in n and "lm_head" not in n:
                muon_params.append(p)
            else:
                aux_params.append(p)
        assert muon_params, "muon: no hidden weight matrices found (n_layer=0?)"
        return Muon(
            muon_params, aux_params, lr=lr, weight_decay=cfg.weight_decay,
            momentum=cfg.muon_momentum, nesterov=cfg.muon_nesterov,
            ns_steps=cfg.muon_ns_steps, betas=tuple(cfg.betas), eps=cfg.eps,
            aux_lr_scale=cfg.muon_aux_lr_scale,
        )
    raise ValueError(f"unknown optimizer name: {cfg.name!r} (sgd|adamw|adam|muon)")


def make_lr_fn(lr_cfg, max_iters):
    """Return get_lr(it) -> learning rate for iteration `it`.

    Supports linear warmup followed by either a cosine decay to `min_lr` or a
    constant LR. Warmup is applied for both schedulers when warmup_iters > 0.
    """
    peak = lr_cfg.learning_rate
    floor = lr_cfg.min_lr
    warmup = lr_cfg.warmup_iters
    sched = lr_cfg.scheduler.lower()

    def get_lr(it):
        if warmup > 0 and it < warmup:
            return peak * (it + 1) / warmup
        if sched == "constant":
            return peak
        if sched == "cosine":
            if it > max_iters:
                return floor
            ratio = (it - warmup) / max(1, (max_iters - warmup))
            coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
            return floor + coeff * (peak - floor)
        raise ValueError(f"unknown scheduler: {lr_cfg.scheduler!r} (cosine|constant)")

    return get_lr
