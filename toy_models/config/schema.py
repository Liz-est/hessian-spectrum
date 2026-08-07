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
    block_type: str = "transformer"  # "transformer" (attn+FFN) | "mlp" (FFN+FFN)
    block_size: int = 128      # context length
    dropout: float = 0.0
    attn_dropout: float = 0.0
    # training objective:
    #   "ce"  -> softmax cross-entropy on integer targets (the default)
    #   "mse" -> mean-squared error between logits and the one-hot target
    #            vector (ignore_index=-1 positions are still dropped)
    #   "mse_rep" -> replication convention: 0.5 * per-position SUM over
    #            classes of the squared error vs one-hot, averaged over
    #            positions (== (V/2) * "mse"; matches RotatedMatrixBigram)
    loss_type: str = "ce"
    # sinusoidal position encoding (vanilla_model only): False -> forward skips
    # adding pos_enc, making the 0-layer model position-independent like
    # simpliest_model while keeping the vanilla_model class / ckpt loading path.
    use_pos_enc: bool = True


# ---------------------------------------------------------------------------
# dataset / batching
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    # dataset directory name under <repo-root>/data/
    dataset: str = "synth_uniform_balanced_V1024"
    batch_size: int = 64       # per-GPU batch; effective batch = batch_size * world_size
    # data on-disk layout / batching:
    #   "dual_stream"    -> train_x.bin + train_y.bin, x and y stored separately
    #                       (the synth bigram data; targets read straight from y).
    #   "nanogpt_shards" -> fineweb_{train,val}_*.bin modded-nanoGPT shards: a
    #                       single uint16 token stream per shard behind a 1024-byte
    #                       header; targets are the inputs shifted by one.
    format: str = "dual_stream"


# ---------------------------------------------------------------------------
# optimizer: `name` selects the torch optimizer; the remaining fields are the
# union of hyper-params any supported optimizer might read. build_optimizer()
# in build.py picks the relevant subset per optimizer, so unused fields (e.g.
# betas for plain SGD) are simply ignored.
# ---------------------------------------------------------------------------
@dataclass
class OptimConfig:
    name: str = "sgd"                       # "sgd" | "adamw" | "adam" | "muon"
    weight_decay: float = 0.1
    momentum: float = 0.9                   # SGD momentum
    nesterov: bool = False                  # SGD nesterov
    betas: Tuple[float, float] = (0.9, 0.95)  # Adam(W) betas; also Muon's aux-AdamW betas
    eps: float = 1e-8                       # Adam(W) epsilon
    grad_clip: float = 1.0                  # 0 disables gradient clipping
    # ---- Muon, Moonlight version (hidden 2D weight matrices; everything else
    # ---- falls back to an internal AdamW group -- see build.Muon) ----
    muon_momentum: float = 0.95             # momentum of the orthogonalized update
    muon_nesterov: bool = True
    muon_ns_steps: int = 5                  # Newton-Schulz iterations
    # LR of the aux AdamW group = scheduled lr * this scale. The Moonlight
    # update is RMS-matched to AdamW (0.2*sqrt(max(d_out,d_in)) scaling), so
    # both groups share the same lr by default; keep 1.0 unless deliberately
    # decoupling the embeds/head LR.
    muon_aux_lr_scale: float = 1.0


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
    # comma-separated submodule names to freeze at init (requires_grad=False
    # on all their params, set BEFORE the DDP wrap and the optimizer build so
    # DDP's reducer skips them and build_optimizer never sees them). Names are
    # resolved with model.get_submodule, so dotted paths work too:
    #   --train.freeze=lm_head              # head only
    #   --train.freeze=lm_head,tok_emb      # head + embedding
    #   --train.freeze=blocks.0             # a whole block
    freeze: str = ""
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
    # how the analyzed token ids are picked for embedding / lm_head blocks:
    #   "first" -> ids 0..max-1 (fine for the synth data, whose ids are already
    #              sorted by descending frequency)
    #   "freq"  -> the max most frequent ids by corpus unigram count, read from
    #              data/<dataset>/token_counts.npy (build it with
    #              compute_fineweb_token_freq.py)
    token_select: str = "first"
    num_bins: int = 64            # log-eigenvalue histogram bins
    seed: int = 1337
    # sub-directory name under toy_models/files/ for eigs/hetero npy + figures
    files_name: str = "vanilla_imbalance_s1-sgd"
    # ---- full-parameter SLQ (analyze_full_spectrum.py) ----
    # if True, submit_sco_vanilla.py appends the SLQ phase to the job, so one
    # submission runs train -> per-unit Hessian -> full-parameter SLQ.
    slq: bool = False
    slq_m: int = 100              # Lanczos steps per probe vector
    slq_num_v: int = 16           # random probe vectors per checkpoint
    slq_n_batches: int = 64       # fixed batches the HVP averages over
    slq_sigma2: float = 1e-5      # gaussian broadening variance for the density
    slq_dtype: str = "fp64"       # HVP/Lanczos precision: "fp64" | "fp32"


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
            block_type=m.block_type, block_size=m.block_size, dropout=m.dropout,
            attn_dropout=m.attn_dropout, loss_type=m.loss_type,
            use_pos_enc=m.use_pos_enc,
        )
