"""
Builders that turn config blocks into live torch objects.

Keeping these here (rather than inline in the trainer) means "which optimizer /
which LR schedule" is fully driven by OptimConfig / LRConfig -- adding a new
optimizer is one branch here, not an edit to the training loop.
"""

import math

import torch


def build_optimizer(params, cfg):
    """Construct a torch optimizer from an OptimConfig.

    `params` is anything torch accepts (model.parameters()). Each branch reads
    only the hyper-params that optimizer actually uses; the rest of OptimConfig
    is ignored, so e.g. `betas` is harmless for plain SGD.
    """
    name = cfg.name.lower()
    lr = 0.0  # real LR is set per-step by the scheduler; start at 0 for warmup
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
    raise ValueError(f"unknown optimizer name: {cfg.name!r} (sgd|adamw|adam)")


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
