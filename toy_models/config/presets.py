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

# "mlp10" model: 5 blocks, each block's attention slot replaced by a second FFN
# (block_type="mlp"), so the model is 10 FFN sub-layers + embed + lm_head, no
# attention. Same d/d_ff as model C.
_MODEL_10FFN = ModelConfig(
    vocab_size=1024, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=5, block_type="mlp", block_size=128,
)

# Layer = 5
_MODEL_l5 = ModelConfig(
    vocab_size=1024, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=5, block_size=128,
)

# 5-layer transformer for the REAL-DATA FineWeb-10B corpus: GPT-2 BPE vocab
# (50257 padded to a multiple of 64) and a 1024-token context. Same d/d_ff/
# n_head as the synth model C, only vocab_size / block_size / n_layer differ.
_MODEL_fw10B = ModelConfig(
    vocab_size=50304, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=5, block_size=1024,
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
    "simpliest_sgd-imbalance": ExperimentConfig(
        name="simpliest_sgd-imbalance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="simpliest_imbalance_s1-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="simpliest_imbalance_s1-sgd"),
    ),

        # ---- simpliest: embedding + lm_head only (n_layer=0), SGD, balance data ----
    "simpliest_sgd-balance": ExperimentConfig(
        name="simpliest_sgd-balance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD),
        data=DataConfig(dataset="synth_uniform_balanced_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="simpliest_balance-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="simpliest_balance-sgd"),
    ),

        # ---- simpliest: embedding + lm_head only (n_layer=0), Adamw, imbalance data ----
    "simpliest_adamw-imbalance": ExperimentConfig(
        name="simpliest_adamw-imbalance",
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

            # ---- simpliest: embedding + lm_head only (n_layer=0), Adamw, balance data ----
    "simpliest_adamw-balance": ExperimentConfig(
        name="simpliest_adamw-balance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD),
        data=DataConfig(dataset="synth_uniform_balanced_V1024", batch_size=64),
       optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="simpliest_balance-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="simpliest_balance-adamw"),
    ),

    # ---- simpliest on the 1B zipf-IMBALANCED synth data, SGD ----
    # 100x more tokens than the 10M sets, so max_iters is bumped 8k -> 80k
    # (~655M tokens seen, well under one epoch => no repeat sampling) and
    # warmup scaled proportionally (2.5% of the schedule, as before).
    "simpliest_imbalance_1B_sgd": ExperimentConfig(
        name="simpliest_imbalance_1B_sgd",
        model=copy.deepcopy(_MODEL_EMBED_HEAD),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024_1B", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=2000),
        train=TrainConfig(max_iters=130000, run_name="simpliest_imbalance_s1_1B-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="simpliest_imbalance_s1_1B-sgd"),
    ),

    # ---- simpliest on the 1B uniform-BALANCED synth data, SGD ----
    "simpliest_balance_1B_sgd": ExperimentConfig(
        name="simpliest_balance_1B_sgd",
        model=copy.deepcopy(_MODEL_EMBED_HEAD),
        data=DataConfig(dataset="synth_uniform_balanced_V1024_1B", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=2000),
        train=TrainConfig(max_iters=130000, run_name="simpliest_balance_1B-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="simpliest_balance_1B-sgd"),
    ),

#   -------------------------------------Layer = 5------------------------------------------

    # ----------------------- SGD imbalance ---------------------
    "layer5-imbalance-s1-sgd": ExperimentConfig(
        name="layer5-imbalance-s1-sgd",
        model=copy.deepcopy(_MODEL_l5),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer5-imbalance-s1-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-imbalance-s1-sgd"),
    ),

    # ----------------------- SGD balance ---------------------
    "layer5-balance-sgd": ExperimentConfig(
        name="layer5-balance-sgd",
        model=copy.deepcopy(_MODEL_l5),
        data=DataConfig(dataset="synth_uniform_balanced_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer5-balance-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-balance-sgd"),
    ),

    # ----------------------- Adamw imbalance ---------------------
    "layer5-imbalance-s1-adamw": ExperimentConfig(
        name="layer5-imbalance-s1-adamw",
        model=copy.deepcopy(_MODEL_l5),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer5-imbalance-s1-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-imbalance-s1-adamw"),
    ),

        # ----------------------- Adam balance ---------------------
    "layer5-balance-adamw": ExperimentConfig(
        name="layer5-balance-adamw",
        model=copy.deepcopy(_MODEL_l5),
        data=DataConfig(dataset="synth_uniform_balanced_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer5-balance-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-balance-adamw"),
    ),

        # ---------------------- Adam balance 1B --------------------
    
    "layer5-balance-1B-adamw": ExperimentConfig(
        name="layer5-balance-1B-adamw",
        model=copy.deepcopy(_MODEL_l5),
        data=DataConfig(dataset="synth_uniform_balanced_V1024_1B", batch_size=64),
        optim=OptimConfig(name="adamw", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=2000),
        train=TrainConfig(max_iters=130000, run_name="layer5-balance-1B-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-balance-1B-adamw"),
    ),

        # ---------------------- Adam imbalance 1B --------------------
    
    "layer5-imbalance-s1-1B-adamw": ExperimentConfig(
        name="layer5-imbalance-s1-1B-adamw",
        model=copy.deepcopy(_MODEL_l5),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024_1B", batch_size=64),
        optim=OptimConfig(name="adamw", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=2000),
        train=TrainConfig(max_iters=130000, run_name="layer5-imbalance-s1-1B-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-imbalance-s1-1B-adamw"),
    ),

#   ------------------------- mlp10: 5 blocks, attn replaced by FFN (no attention) -----------
#   10 FFN sub-layers + embed + lm_head. 10M data, max_iters 8k -> 20k for the
#   larger model (warmup 500 = 2.5% of the schedule).

    # ----------------------- SGD imbalance ---------------------
    "mlp10_sgd-imbalance": ExperimentConfig(
        name="mlp10_sgd-imbalance",
        model=copy.deepcopy(_MODEL_10FFN),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=500),
        train=TrainConfig(max_iters=20000, run_name="mlp10_imbalance_s1-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mlp10_imbalance_s1-sgd"),
    ),

    # ----------------------- SGD balance ---------------------
    "mlp10_sgd-balance": ExperimentConfig(
        name="mlp10_sgd-balance",
        model=copy.deepcopy(_MODEL_10FFN),
        data=DataConfig(dataset="synth_uniform_balanced_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=500),
        train=TrainConfig(max_iters=20000, run_name="mlp10_balance-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mlp10_balance-sgd"),
    ),

    # ----------------------- AdamW imbalance ---------------------
    "mlp10_adamw-imbalance": ExperimentConfig(
        name="mlp10_adamw-imbalance",
        model=copy.deepcopy(_MODEL_10FFN),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=500),
        train=TrainConfig(max_iters=20000, run_name="mlp10_imbalance_s1-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mlp10_imbalance_s1-adamw"),
    ),

    # ----------------------- AdamW balance ---------------------
    "mlp10_adamw-balance": ExperimentConfig(
        name="mlp10_adamw-balance",
        model=copy.deepcopy(_MODEL_10FFN),
        data=DataConfig(dataset="synth_uniform_balanced_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=500),
        train=TrainConfig(max_iters=20000, run_name="mlp10_balance-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mlp10_balance-adamw"),
    ),


#   ------------------------- fineweb10B: 5-layer transformer on REAL data -----------
#   GPT-2 BPE (vocab 50304), 1024-token context, AdamW. Training budget follows
#   train_vanilla_transformer_fineweb10B.py (20k iters, bs=32, warmup=400,
#   lr 6e-4 -> 6e-5). Data lives in <repo-root>/data/fineweb10B/ as modded-nanoGPT
#   single-stream shards, so format="nanogpt_shards". 9-checkpoint schedule.

    "layer5-fineweb10B-adamw": ExperimentConfig(
        name="layer5-fineweb10B-adamw",
        model=copy.deepcopy(_MODEL_fw10B),
        data=DataConfig(dataset="fineweb10B", format="nanogpt_shards", batch_size=32),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=6e-5,
                    warmup_iters=400),
        train=TrainConfig(max_iters=20000, run_name="layer5-fineweb10B-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-fineweb10B-adamw"),
    ),

    "layer5-fineweb10B-sgd": ExperimentConfig(
        name="layer5-fineweb10B-sgd",
        model=copy.deepcopy(_MODEL_fw10B),
        data=DataConfig(dataset="fineweb10B", format="nanogpt_shards", batch_size=32),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=6e-5,
                    warmup_iters=400),
        train=TrainConfig(max_iters=20000, run_name="layer5-fineweb10B-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-fineweb10B-sgd"),
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
