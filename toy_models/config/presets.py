"""
Active named experiment presets.

Keep this registry limited to the current controlled experiment. Historical
Transformer, FineWeb, freeze, and sweep configurations are retained in
legacy_presets.py but are intentionally not visible to config.load().
"""

import copy

from .schema import (
    AnalyzeConfig,
    DataConfig,
    ExperimentConfig,
    LRConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)


# Zero-layer, position-independent bilinear model trained against one-hot
# targets with MSE. Nonzero embedding noise escapes the zero/zero stationary
# point; the LM-head matrix starts exactly at zero.
_MODEL_FULLBATCH = ModelConfig(
    vocab_size=1024,
    n_embd=192,
    n_head=6,
    head_dim=32,
    n_ffn=1024,
    n_layer=0,
    block_size=128,
    loss_type="mse",
    use_pos_enc=False,
    linear_bias=False,
    tok_emb_init_mean=0.0,
    tok_emb_init_std=0.02,
    lm_head_init_mean=0.0,
    lm_head_init_std=0.0,
)

_DATASET = "synth_shuffled_x1_y0_2p17train_2p14val_V1024"

_CKPT_9 = {
    "init": 0.0,
    "p10": 0.10,
    "p25": 0.25,
    "p40": 0.40,
    "p50": 0.50,
    "p60": 0.60,
    "p75": 0.75,
    "p85": 0.85,
    "p100": 1.0,
}


EXPERIMENTS = {
    "fullbatch-mse0-shuffled-2p17-gd": ExperimentConfig(
        name="fullbatch-mse0-shuffled-2p17-gd",
        model=copy.deepcopy(_MODEL_FULLBATCH),
        data=DataConfig(dataset=_DATASET, batch_size=64),
        optim=OptimConfig(
            name="sgd",
            momentum=0.0,
            weight_decay=0.0,
            grad_clip=0,
        ),
        lr=LRConfig(
            scheduler="constant",
            learning_rate=1e-3,
            warmup_iters=0,
        ),
        train=TrainConfig(
            max_iters=200,
            run_name="fullbatch-mse0-shuffled-2p17-gd",
            ckpt_fracs=dict(_CKPT_9),
        ),
        analyze=AnalyzeConfig(
            files_name="fullbatch-mse0-shuffled-2p17-gd",
        ),
    ),
    "fullbatch-mse0-shuffled-2p17-adam": ExperimentConfig(
        name="fullbatch-mse0-shuffled-2p17-adam",
        model=copy.deepcopy(_MODEL_FULLBATCH),
        data=DataConfig(dataset=_DATASET, batch_size=64),
        optim=OptimConfig(
            name="adam",
            betas=(0.0, 0.99),
            eps=1e-8,
            weight_decay=0.0,
            grad_clip=0,
        ),
        lr=LRConfig(
            scheduler="constant",
            learning_rate=1.5e-3,
            warmup_iters=0,
        ),
        train=TrainConfig(
            max_iters=200,
            run_name="fullbatch-mse0-shuffled-2p17-adam",
            ckpt_fracs=dict(_CKPT_9),
        ),
        analyze=AnalyzeConfig(
            files_name="fullbatch-mse0-shuffled-2p17-adam",
        ),
    ),
}


# No implicit named preset: callers either select one of the two experiments
# above or receive a schema-default placeholder for CLI overrides.
DEFAULT = ""


def get(name):
    if name not in EXPERIMENTS:
        raise KeyError(
            f"unknown active experiment {name!r}; "
            f"known: {sorted(EXPERIMENTS)}"
        )
    return copy.deepcopy(EXPERIMENTS[name])
