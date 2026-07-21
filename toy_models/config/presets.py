"""
Named experiment presets.

Each entry in EXPERIMENTS is a fully-specified ExperimentConfig. `config.load(
name)` returns a deep copy so callers can mutate freely (e.g. via CLI
overrides) without touching the registry.

To add an experiment: copy an existing block, give it a unique key, and change
only the fields that differ. Keep run_name / files_name in sync with the key so
each experiment writes to its own runs/ and files/ sub-directory.

The `imbalance_s1_sgd` preset reproduces the settings the scripts hard-coded
before the config refactor (SGD, lr 6e-4, cosine decay, the 9-checkpoint
schedule), so an unspecified `load()` behaves exactly as the old defaults did.
"""

import copy

from .schema import (ExperimentConfig, ModelConfig, DataConfig, OptimConfig,
                     LRConfig, TrainConfig, AnalyzeConfig)

# model C shape is shared across the synth presets below
_MODEL_C = ModelConfig(
    vocab_size=1024, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=1, block_size=128,
)

# "simpliest" model: embedding + lm_head only, no transformer block
# (simpliest_model.py builds config.n_layer blocks, so n_layer=0 => none).
_MODEL_EMBED_HEAD = ModelConfig(
    vocab_size=1024, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=0, block_size=128,
)

_CKPT_9 = {"init": 0.0, "p10": 0.10, "p25": 0.25, "p40": 0.40, "p50": 0.50,
           "p60": 0.60, "p75": 0.75, "p85": 0.85, "p100": 1.0}


EXPERIMENTS = {
    # ---- current default: SGD on the zipf-imbalanced V=1024 synth data ----
    "imbalance_s1_sgd": ExperimentConfig(
        name="imbalance_s1_sgd",
        model=copy.deepcopy(_MODEL_C),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="vanilla_imbalance_s1-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="vanilla_imbalance_s1-sgd"),
    ),

    # ---- AdamW variant, same data/schedule (optimizer comparison) ----
    "imbalance_s1_adamw": ExperimentConfig(
        name="imbalance_s1_adamw",
        model=copy.deepcopy(_MODEL_C),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="vanilla_imbalance_s1-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="vanilla_imbalance_s1-adamw"),
    ),

    # ---- simpliest: embedding + lm_head only (n_layer=0), SGD, same data ----
    "simpliest_sgd": ExperimentConfig(
        name="simpliest_sgd",
        model=copy.deepcopy(_MODEL_EMBED_HEAD),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="simpliest_imbalance_s1-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="simpliest_imbalance_s1-sgd"),
    ),

        # ---- simpliest: embedding + lm_head only (n_layer=0), Adamw, same data ----
    "simpliest_adamw": ExperimentConfig(
        name="simpliest_adamw",
        model=copy.deepcopy(_MODEL_EMBED_HEAD),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
       optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="simpliest_imbalance_s1-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="simpliest_imbalance_s1-adamw"),
    ),
}

# the preset load() falls back to when no name is given
DEFAULT = "imbalance_s1_sgd"


def get(name):
    if name not in EXPERIMENTS:
        raise KeyError(
            f"unknown experiment {name!r}; known: {sorted(EXPERIMENTS)}"
        )
    return copy.deepcopy(EXPERIMENTS[name])
