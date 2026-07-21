"""
Typed configuration schema for the vanilla_transformer toy experiments.

Everything a run needs -- model shape, dataset, optimizer + its hyper-params,
LR schedule, training loop, checkpoint schedule, and the Hessian-analysis knobs
-- lives here as grouped @dataclass blocks, mirroring the style of
ToyVanillaConfig in vanilla_model.py.

An ExperimentConfig bundles all groups. Named presets live in presets.py;
config.load("<name>") returns one, and CLI `--group.key=value` flags override
individual fields (see config/__init__.py).

Design notes
------------
* All fields are plain scalars / small containers so a config is trivially
  serialisable (asdict) into a checkpoint or a summary JSON.
* `ckpt_fracs` is the single source of truth for the checkpoint schedule; both
  the trainer (which *writes* ckpt_<tag>.pt) and the analyzer (which *reads*
  them) derive their tags from the same ExperimentConfig, so they can no longer
  drift apart the way the old ckpt_fracs / TAGS pair could.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# model shape (a superset of ToyVanillaConfig's trainable-shape fields)
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 1024
    n_embd: int = 192          # hidden size d
    n_head: int = 6            # attention heads h
    head_dim: int = 32         # d_head (n_head * head_dim == n_embd here)
    n_ffn: int = 1024          # FFN inner size d_ff
    n_layer: int = 1           # single-layer decoder
    block_size: int = 128      # context length
    dropout: float = 0.0
    attn_dropout: float = 0.0


# ---------------------------------------------------------------------------
# dataset / batching
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    # dataset directory name under <repo-root>/data/
    dataset: str = "synth_uniform_balanced_V1024"
    batch_size: int = 64       # per-GPU batch; effective batch = batch_size * world_size


# ---------------------------------------------------------------------------
# optimizer: `name` selects the torch optimizer; the remaining fields are the
# union of hyper-params any supported optimizer might read. build_optimizer()
# in build.py picks the relevant subset per optimizer, so unused fields (e.g.
# betas for plain SGD) are simply ignored.
# ---------------------------------------------------------------------------
@dataclass
class OptimConfig:
    name: str = "sgd"                       # "sgd" | "adamw" | "adam"
    weight_decay: float = 0.1
    momentum: float = 0.9                   # SGD momentum
    nesterov: bool = False                  # SGD nesterov
    betas: Tuple[float, float] = (0.9, 0.95)  # Adam(W) betas
    eps: float = 1e-8                       # Adam(W) epsilon
    grad_clip: float = 1.0                  # 0 disables gradient clipping


# ---------------------------------------------------------------------------
# learning-rate schedule
# ---------------------------------------------------------------------------
@dataclass
class LRConfig:
    scheduler: str = "cosine"     # "cosine" | "constant"
    learning_rate: float = 6e-4   # peak LR
    min_lr: float = 3e-5          # floor for the cosine tail
    warmup_iters: int = 200       # linear warmup length (0 disables warmup)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    max_iters: int = 8000
    eval_interval: int = 200
    eval_iters: int = 100
    log_interval: int = 20
    seed: int = 1337
    # sub-directory name under toy_models/runs/ for checkpoints + loss curve
    run_name: str = "vanilla_imbalance_s1-sgd"
    # checkpoint at these fractions of training; keys are the tags used as
    # ckpt_<tag>.pt filenames and as the analyzer's per-checkpoint labels.
    ckpt_fracs: Dict[str, float] = field(default_factory=lambda: {
        "init": 0.0, "p10": 0.10, "p25": 0.25, "p40": 0.40, "p50": 0.50,
        "p60": 0.60, "p75": 0.75, "p85": 0.85, "p100": 1.0,
    })


# ---------------------------------------------------------------------------
# Hessian / heterogeneity analysis
# ---------------------------------------------------------------------------
@dataclass
class AnalyzeConfig:
    batch_size: int = 32          # curvature batch size (independent of training)
    n_batches: int = 20           # curvature batches accumulated per layer
    max_classes: int = 256        # per-token lm_head blocks to compute (<= vocab_size)
    max_tokens: int = 256         # per-token embedding blocks to compute (<= vocab_size)
    num_bins: int = 64            # log-eigenvalue histogram bins
    seed: int = 1337
    # sub-directory name under toy_models/files/ for eigs/hetero npy + figures
    files_name: str = "vanilla_imbalance_s1-sgd"


# ---------------------------------------------------------------------------
# top-level bundle
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    lr: LRConfig = field(default_factory=LRConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    analyze: AnalyzeConfig = field(default_factory=AnalyzeConfig)

    def ckpt_iters(self) -> Dict[str, int]:
        """Map each checkpoint tag to its integer iteration (frac * max_iters)."""
        mi = self.train.max_iters
        iters = {tag: round(frac * mi) for tag, frac in self.train.ckpt_fracs.items()}
        # guard against two fractions rounding onto the same iteration, which
        # would make one checkpoint silently overwrite another.
        if len(set(iters.values())) != len(iters):
            raise ValueError(f"ckpt_fracs collide at max_iters={mi}: {iters}")
        return iters

    def to_model_config(self):
        """Build the ToyVanillaConfig the model class expects from ModelConfig."""
        from vanilla_model import ToyVanillaConfig
        m = self.model
        return ToyVanillaConfig(
            vocab_size=m.vocab_size, n_embd=m.n_embd, n_head=m.n_head,
            head_dim=m.head_dim, n_ffn=m.n_ffn, n_layer=m.n_layer,
            block_size=m.block_size, dropout=m.dropout,
            attn_dropout=m.attn_dropout,
        )
