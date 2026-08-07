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

_MODEL_EMBED_HEAD_POS0 = ModelConfig(
    vocab_size=1024, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=0, block_size=128, use_pos_enc=False
)

# same 0-layer shape but trained with MSE-vs-one-hot instead of cross-entropy
# (run through vanilla_model.py, which reads loss_type in its forward()).
_MODEL_EMBED_HEAD_MSE = ModelConfig(
    vocab_size=1024, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=0, block_size=128, loss_type="mse",
)

# same as _MODEL_EMBED_HEAD_MSE but with the sinusoidal pos_enc disabled
# (use_pos_enc=False -> vanilla_model.forward skips the pos_enc add), so the
# 0-layer model is purely token->logits with no position dependence. Removes
# the Var_t(sum pe[t]) ~= 109 irreducible-loss floor seen in the frozen-lm_head
# (all-ones, rank-1) runs, where every class shares one logit and the position
# term cannot be cancelled per position.
_MODEL_EMBED_HEAD_MSE_POS0 = ModelConfig(
    vocab_size=1024, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=0, block_size=128, loss_type="mse",
    use_pos_enc=False,
)

# replication of signal frequency
_MODEL_REPLICATION = ModelConfig(
    vocab_size=10000, n_embd=10000, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=0, block_size=128, loss_type="mse_rep",
    use_pos_enc=False,
)

# same shape but the original "mse" convention (mean over classes), matching
# the earlier REP-* bigram runs -- used by the SGD lr sweep below.
_MODEL_REPLICATION_MSE = ModelConfig(
    vocab_size=10000, n_embd=10000, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=0, block_size=128, loss_type="mse",
    use_pos_enc=False,
)

_MODEL_l5_MSE = ModelConfig(
    vocab_size=1024, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=5, block_size=128, loss_type="mse",
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

# Layer = 20
_MODEL_l20 = ModelConfig(
    vocab_size=1024, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=20, block_size=128,
)

# 5-layer transformer for the REAL-DATA FineWeb-10B corpus: GPT-2 BPE vocab
# (50257 padded to a multiple of 64) and a 1024-token context. Same d/d_ff/
# n_head as the synth model C, only vocab_size / block_size / n_layer differ.
_MODEL_fw10B_l5 = ModelConfig(
    vocab_size=50304, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=5, block_size=1024,
)

_MODEL_fw10B_l20 = ModelConfig(
    vocab_size=50304, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=20, block_size=1024,
)

# 1-layer variant, matching the legacy pre-preset run vanilla_fineweb10B-adamw
# (train_vanilla_transformer_fineweb10B.py: same shape, n_layer=1).
_MODEL_fw10B_l1 = ModelConfig(
    vocab_size=50304, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=1, block_size=1024,
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

    # ---- 1-layer transformer on the BALANCED V=1024 synth data (SGD) ----
    # mirrors imbalance_s1_sgd, only the dataset differs (balance vs zipf).
    "balance_sgd": ExperimentConfig(
        name="balance_sgd",
        model=copy.deepcopy(_MODEL_C),
        data=DataConfig(dataset="synth_uniform_balanced_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="vanilla_balance-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="vanilla_balance-sgd"),
    ),

    # ---- 1-layer transformer on the BALANCED V=1024 synth data (AdamW) ----
    "balance_adamw": ExperimentConfig(
        name="balance_adamw",
        model=copy.deepcopy(_MODEL_C),
        data=DataConfig(dataset="synth_uniform_balanced_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="vanilla_balance-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="vanilla_balance-adamw"),
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

    # ---- 0-layer transformer trained with MSE loss (vs one-hot targets) ----
    # embed + lm_head only, run via vanilla_model.py (loss_type="mse").
    # 4 presets: {imbalance, balance} x {sgd, adamw}.
    "mse0_sgd-imbalance": ExperimentConfig(
        name="mse0_sgd-imbalance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="mse0_imbalance_s1-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0_imbalance_s1-sgd"),
    ),

    "mse0_sgd-balance": ExperimentConfig(
        name="mse0_sgd-balance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE),
        data=DataConfig(dataset="synth_uniform_balanced_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="mse0_balance-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0_balance-sgd"),
    ),

    "mse0_adamw-imbalance": ExperimentConfig(
        name="mse0_adamw-imbalance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="mse0_imbalance_s1-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0_imbalance_s1-adamw"),
    ),

    "mse0_adamw-balance": ExperimentConfig(
        name="mse0_adamw-balance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE),
        data=DataConfig(dataset="synth_uniform_balanced_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="mse0_balance-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0_balance-adamw"),
    ),

    # ---- simpliest on the 1B zipf-IMBALANCED synth data, SGD ----
    # 100x more tokens than the 10M sets, so max_iters is bumped 8k -> 130k
    # (130k * 64 * 128 ~= 1.065B tokens ~= 1.06 epochs => ~one pass, no heavy
    # repeat sampling) with warmup 2000 (~1.5% of the schedule; note this is
    # lower than the 2.5% used by the 10M presets -- warmup was NOT rescaled
    # when max_iters moved from the original 80k plan to 130k).
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
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
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
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
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
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
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
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=2000),
        train=TrainConfig(max_iters=130000, run_name="layer5-imbalance-s1-1B-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-imbalance-s1-1B-adamw"),
    ),


# ---------------------------------------- Layer = 20 ------------------------------------------

# ----------------------- SGD imbalance ---------------------
    "layer20-imbalance-s1-sgd": ExperimentConfig(
        name="layer20-imbalance-s1-sgd",
        model=copy.deepcopy(_MODEL_l20),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer20-imbalance-s1-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer20-imbalance-s1-sgd"),
    ),

    # ----------------------- SGD balance ---------------------
    "layer20-balance-sgd": ExperimentConfig(
        name="layer20-balance-sgd",
        model=copy.deepcopy(_MODEL_l20),
        data=DataConfig(dataset="synth_uniform_balanced_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer20-balance-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer20-balance-sgd"),
    ),

    # ----------------------- Adamw imbalance ---------------------
    "layer20-imbalance-s1-adamw": ExperimentConfig(
        name="layer20-imbalance-s1-adamw",
        model=copy.deepcopy(_MODEL_l20),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer20-imbalance-s1-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer20-imbalance-s1-adamw"),
    ),

        # ----------------------- Adam balance ---------------------
    "layer20-balance-adamw": ExperimentConfig(
        name="layer20-balance-adamw",
        model=copy.deepcopy(_MODEL_l20),
        data=DataConfig(dataset="synth_uniform_balanced_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer20-balance-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer20-balance-adamw"),
    ),

        # ---------------------- Adam balance 1B --------------------
    
    "layer20-balance-1B-adamw": ExperimentConfig(
        name="layer20-balance-1B-adamw",
        model=copy.deepcopy(_MODEL_l20),
        data=DataConfig(dataset="synth_uniform_balanced_V1024_1B", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=2000),
        train=TrainConfig(max_iters=130000, run_name="layer20-balance-1B-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer20-balance-1B-adamw"),
    ),

        # ---------------------- Adam imbalance 1B --------------------
    
    "layer20-imbalance-s1-1B-adamw": ExperimentConfig(
        name="layer20-imbalance-s1-1B-adamw",
        model=copy.deepcopy(_MODEL_l20),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024_1B", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5,
                    warmup_iters=2000),
        train=TrainConfig(max_iters=130000, run_name="layer20-imbalance-s1-1B-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer20-imbalance-s1-1B-adamw"),
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
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="mlp10_imbalance_s1-sgd",
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
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="mlp10_balance-sgd",
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
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="mlp10_imbalance_s1-adamw",
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
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="mlp10_balance-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mlp10_balance-adamw"),
    ),


#   ------------------------- fineweb10B: 5-layer/20 layer transformer on REAL data -----------
#   GPT-2 BPE (vocab 50304), 1024-token context, AdamW. Training budget follows
#   train_vanilla_transformer_fineweb10B.py (20k iters, bs=32, warmup=400,
#   lr 6e-4 -> 6e-5). Data lives in <repo-root>/data/fineweb10B/ as modded-nanoGPT
#   single-stream shards, so format="nanogpt_shards". 9-checkpoint schedule.

    "layer5-fineweb10B-adamw": ExperimentConfig(
        name="layer5-fineweb10B-adamw",
        model=copy.deepcopy(_MODEL_fw10B_l5),
        data=DataConfig(dataset="fineweb10B", format="nanogpt_shards", batch_size=32),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=6e-5,
                    warmup_iters=400),
        train=TrainConfig(max_iters=20000, run_name="layer5-fineweb10B-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-fineweb10B-adamw",
                               max_classes=1024, max_tokens=1024,
                               token_select="freq"),
    ),

    "layer5-fineweb10B-sgd": ExperimentConfig(
        name="layer5-fineweb10B-sgd",
        model=copy.deepcopy(_MODEL_fw10B_l5),
        data=DataConfig(dataset="fineweb10B", format="nanogpt_shards", batch_size=32),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=6e-5,
                    warmup_iters=400),
        train=TrainConfig(max_iters=20000, run_name="layer5-fineweb10B-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-fineweb10B-sgd",
                               max_classes=1024, max_tokens=1024,
                               token_select="freq"),
    ),

    "layer20-fineweb10B-adamw": ExperimentConfig(
        name="layer20-fineweb10B-adamw",
        model=copy.deepcopy(_MODEL_fw10B_l20),
        data=DataConfig(dataset="fineweb10B", format="nanogpt_shards", batch_size=32),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=6e-5,
                    warmup_iters=400),
        train=TrainConfig(max_iters=20000, run_name="layer20-fineweb10B-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer20-fineweb10B-adamw",
                               max_classes=1024, max_tokens=1024,
                               token_select="freq"),
    ),

    "layer20-fineweb10B-sgd": ExperimentConfig(
        name="layer20-fineweb10B-sgd",
        model=copy.deepcopy(_MODEL_fw10B_l20),
        data=DataConfig(dataset="fineweb10B", format="nanogpt_shards", batch_size=32),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=6e-5,
                    warmup_iters=400),
        train=TrainConfig(max_iters=20000, run_name="layer20-fineweb10B-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer20-fineweb10B-sgd",
                               max_classes=1024, max_tokens=1024,
                               token_select="freq"),
    ),

    "vanilla_fineweb10B-adamw": ExperimentConfig(
        name="vanilla_fineweb10B-adamw",
        model=copy.deepcopy(_MODEL_fw10B_l1),
        data=DataConfig(dataset="fineweb10B", format="nanogpt_shards", batch_size=32),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1,
                          grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=6e-5,
                    warmup_iters=400),
        train=TrainConfig(max_iters=20000, run_name="vanilla_fineweb10B-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="vanilla_fineweb10B-adamw",
                               max_classes=1024, max_tokens=1024,
                               token_select="freq"),
    ),

    "vanilla_fineweb10B-sgd": ExperimentConfig(
        name="vanilla_fineweb10B-sgd",
        model=copy.deepcopy(_MODEL_fw10B_l1),
        data=DataConfig(dataset="fineweb10B", format="nanogpt_shards", batch_size=32),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=6e-5,
                    warmup_iters=400),
        train=TrainConfig(max_iters=20000, run_name="vanilla_fineweb10B-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="vanilla_fineweb10B-sgd",
                               max_classes=1024, max_tokens=1024,
                               token_select="freq"),
    ),

#-----------------------Layer5 + MSE loss----------------------------

    # imbalance data + SGD
    "layer5-mse-imbalance-s1-sgd": ExperimentConfig(
        name="layer5-mse-imbalance-s1-sgd",
        model=copy.deepcopy(_MODEL_l5_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.25, min_lr=0.0125,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer5-mse-imbalance-s1-sgd",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-mse-imbalance-s1-sgd"),
    ),

    # imbalance data + SGD + no gradient clipping
    "layer5-mse-imbalance-s1-sgd-gradclip0": ExperimentConfig(
        name="layer5-mse-imbalance-s1-sgd-gradclip0",
        model=copy.deepcopy(_MODEL_l5_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.1, min_lr=0.005,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer5-mse-imbalance-s1-sgd-gradclip0",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-mse-imbalance-s1-sgd-gradclip0"),
    ),

    # imbalance data + AdamW
    "layer5-mse-imbalance-s1-adamw": ExperimentConfig(
        name="layer5-mse-imbalance-s1-adamw",
        model=copy.deepcopy(_MODEL_l5_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-4, min_lr=0.075e-4,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer5-mse-imbalance-s1-adamw",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-mse-imbalance-s1-adamw"),
    ),

    # imbalance data + Muon
    "layer5-mse-imbalance-s1-muon": ExperimentConfig(
        name="layer5-mse-imbalance-s1-muon",
        model=copy.deepcopy(_MODEL_l5_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="muon", weight_decay=0.1, grad_clip=1.0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-4, min_lr=0.075e-4,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer5-mse-imbalance-s1-muon",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-mse-imbalance-s1-muon"),
    ),

    # imbalance data + Muon + no gradient clipping
    "layer5-mse-imbalance-s1-muon": ExperimentConfig(
        name="layer5-mse-imbalance-s1-muon-gradclip0",
        model=copy.deepcopy(_MODEL_l5_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="muon", weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-4, min_lr=0.075e-4,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer5-mse-imbalance-s1-muon",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-mse-imbalance-s1-muon"),
    ),

#------------------------Freeze Test(delete the position encoding)-----------------------------------
# linear model (only embedding + lm_head) + mes loss + imbalance data (initialize = 1)
    # ---------freeze embedding + SGD lr = 0.1---------------------------
    "mse0-pos0-frozen_embd-sgd-lr0p1-imb-init1G": ExperimentConfig(
        name="mse0-pos0-frozen_embd-sgd-lr0p1-imb-init1G",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.1, min_lr=0.005,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=8000, run_name="mse0-pos0-frozen_embd-sgd-lr0p1-imb-init1G",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_embd-sgd-lr0p1-imb-init1G"),
    ),
    # ---------freeze embedding + AdamW lr = 1.5e-3--------------------------
    "mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-init1G": ExperimentConfig(
        name="mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-init1G",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-3, min_lr=0.075e-3,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=8000, run_name="mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-init1G",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-init1G"),
    ),

    # ---------freeze lm_head + SGD lr = 0.05------------------------------
    "mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G": ExperimentConfig(
        name="mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.05, min_lr=0.0025,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=8000, run_name="mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G"),
    ),

    # ---------freeze lm_head + AdamW lr = 1.5e-3----------------------------
    "mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G": ExperimentConfig(
        name="mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-3, min_lr=0.075e-3,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=8000, run_name="mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G"),
    ),

        # ---------freeze lm_head + AdamW lr = 1.5e-2----------------------------
    "mse0-pos0-frozen_lmhead-adamw-lr1p5e-2-imb-init1G": ExperimentConfig(
        name="mse0-pos0-frozen_lmhead-adamw-lr1p5e-2-imb-init1G",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-2, min_lr=0.075e-2,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=8000, run_name="mse0-pos0-frozen_lmhead-adamw-lr1p5e-2-imb-init1G",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_lmhead-adamw-lr1p5e-2-imb-init1G"),
    ),

            # ---------freeze lm_head + AdamW lr = 2e-3----------------------------
    "mse0-pos0-frozen_lmhead-adamw-lr2e-3-imb-init1G": ExperimentConfig(
        name="mse0-pos0-frozen_lmhead-adamw-lr2e-3-imb-init1G",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=2e-3, min_lr=0.1e-3,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=8000, run_name="mse0-pos0-frozen_lmhead-adamw-lr2e-3-imb-init1G",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_lmhead-adamw-lr2e-3-imb-init1G"),
    ),
    # -----------------------------1500 iter------------------------------------
    # ---------freeze embedding + SGD lr = 0.1---------------------------
    "mse0-pos0-frozen_embd-sgd-lr0p1-imb-initG-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_embd-sgd-lr0p1-imb-initG-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.1, min_lr=0.005,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=1500, run_name="mse0-pos0-frozen_embd-sgd-lr0p1-imb-initG-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_embd-sgd-lr0p1-imb-initG-iter1500"),
    ),
    # ---------freeze embedding + AdamW lr = 1.5e-3--------------------------
    "mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-initG-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-initG-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-3, min_lr=0.075e-3,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=1500, run_name="mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-initG-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-initG-iter1500"),
    ),

    # ---------freeze lm_head + SGD lr = 0.05------------------------------
    "mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-initG-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-initG-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.05, min_lr=0.0025,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=1500, run_name="mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-initG-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-initG-iter1500"),
    ),

    # ---------freeze lm_head + AdamW lr = 1.5e-3----------------------------
    "mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-initG-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-initG-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-3, min_lr=0.075e-3,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=1500, run_name="mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-initG-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-initG-iter1500"),
    ),

        # ---------freeze lm_head + AdamW lr = 6e-3----------------------------
    "mse0-pos0-frozen_lmhead-adamw-lr6e-3-imb-initG-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_lmhead-adamw-lr6e-3-imb-initG-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-3, min_lr=3e-4,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=1500, run_name="mse0-pos0-frozen_lmhead-adamw-lr6e-3-imb-initG-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_lmhead-adamw-lr6e-3-imb-initG-iter1500"),
    ),

            # ---------freeze embedding + AdamW lr = 6e-3----------------------------
    "mse0-pos0-frozen_embd-adamw-lr6e-3-imb-initG-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_embd-adamw-lr6e-3-imb-initG-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-3, min_lr=3e-4,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=1500, run_name="mse0-pos0-frozen_embd-adamw-lr6e-3-imb-initG-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_embd-adamw-lr6e-3-imb-initG-iter1500"),
    ),

            # ---------freeze lm_head + AdamW lr = 9e-3----------------------------
    "mse0-pos0-frozen_lmhead-adamw-lr9e-3-imb-initG-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_lmhead-adamw-lr9e-3-imb-initG-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=9e-3, min_lr=4.5e-4,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=1500, run_name="mse0-pos0-frozen_lmhead-adamw-lr9e-3-imb-initG-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_lmhead-adamw-lr9e-3-imb-initG-iter1500"),
    ),

            # ---------freeze embedding + AdamW lr = 9e-3----------------------------
    "mse0-pos0-frozen_embd-adamw-lr9e-3-imb-initG-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_embd-adamw-lr9e-3-imb-initG-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=9e-3, min_lr=4.5e-4,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=1500, run_name="mse0-pos0-frozen_embd-adamw-lr9e-3-imb-initG-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_embd-adamw-lr9e-3-imb-initG-iter1500"),
    ),

    # ---- 0-layer MSE, pos_enc off, FULL-BATCH on 10M imbalance data ----
    # per-GPU bs 9766 x 8 ranks x block 128 = 10,000,384 tok/step ~= the whole
    # 10M train set per iteration (windows sampled WITH replacement, so this is
    # a fresh ~1-epoch sample each step, not a fixed deterministic batch).
    "mse0-pos0-frz_lmhead-fullbs-sgd-imb": ExperimentConfig(
        name="mse0-pos0-frz_lmhead-fullbs-sgd-imb",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=9766),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.05, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze = "lm_head", max_iters=500, eval_iters=2,
                          run_name="mse0-pos0-frz_lmhead-fullbs-sgd-imb",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frz_lmhead-fullbs-sgd-imb"),
    ),

    "mse0-pos0-frz_lmhead-fullbs-adamw-imb": ExperimentConfig(
        name="mse0-pos0-frz_lmhead-fullbs-adamw-imb",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=9766),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8,
                          weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-3, min_lr=3e-4,
                    warmup_iters=50),
        train=TrainConfig(freeze = "lm_head", max_iters=500, eval_iters=2,
                          run_name="mse0-pos0-frz_lmhead-fullbs-adamw-imb",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frz_lmhead-fullbs-adamw-imb"),
    ),

    #-----------------------Change to ce loss-------------------------
    "pos0-frz_lmhead-fullbs-sgd-imb": ExperimentConfig(
        name="mse0-pos0-frz_lmhead-fullbs-sgd-imb",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=9766),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.05, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze = "lm_head", max_iters=500, eval_iters=2,
                          run_name="pos0-frz_lmhead-fullbs-sgd-imb",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="pos0-frz_lmhead-fullbs-sgd-imb"),
    ),

    "pos0-frz_lmhead-fullbs-adamw-imb": ExperimentConfig(
        name="mse0-pos0-frz_lmhead-fullbs-adamw-imb",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=9766),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8,
                          weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-3, min_lr=3e-4,
                    warmup_iters=50),
        train=TrainConfig(freeze = "lm_head", max_iters=500, eval_iters=2,
                          run_name="pos0-frz_lmhead-fullbs-adamw-imb",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="pos0-frz_lmhead-fullbs-adamw-imb"),
    ),

    # -----------------------------Replication of Signal Frequency-----------------------------
    "REP-mse0-pos0-frz_embd-fullbs-sgd-imb-initG02": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-sgd-imb-initG02",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=0.05, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-sgd-imb-initG02",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-sgd-imb-initG02"),
    ),

    "REP-mse0-pos0-frz_embd-fullbs-adam-imb-initG02": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-adam-imb-initG02",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                          weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=6e-3, min_lr=3e-4,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-adam-imb-initG02",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-adam-imb-initG02"),
    ), 

    "REP-mse0-pos0-frz_embd-fullbs-adam-lr2e-3-imb-initG02": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-adam-lr2e-3-imb-initG02",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                          weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=2e-3, min_lr=3e-4,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr2e-3-imb-initG02",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr2e-3-imb-initG02"),
    ),

    "REP-mse0-pos0-frz_embd-fullbs-adam-lr1.5e-3-imb-initG02": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-adam-lr1.5e-3-imb-initG02",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                          weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=1.5e-3, min_lr=3e-4,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr1.5e-3-imb-initG02",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr1.5e-3-imb-initG02"),
    ), 

    "REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-4-imb-initG02": ExperimentConfig(
            name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-4-imb-initG02",
            model=copy.deepcopy(_MODEL_REPLICATION),
            data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
            optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                              weight_decay=0.0, grad_clip=0),
            lr=LRConfig(scheduler="constant", learning_rate=6e-4, min_lr=3e-4,
                        warmup_iters=50),
            train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                              run_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-4-imb-initG02",
                              ckpt_fracs=dict(_CKPT_9)),
            analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-4-imb-initG02"),
        ), 

    "REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-5-imb-initG02": ExperimentConfig(
            name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-5-imb-initG02",
            model=copy.deepcopy(_MODEL_REPLICATION),
            data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
            optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                              weight_decay=0.0, grad_clip=0),
            lr=LRConfig(scheduler="constant", learning_rate=6e-5, min_lr=3e-4,
                        warmup_iters=50),
            train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                              run_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-5-imb-initG02",
                              ckpt_fracs=dict(_CKPT_9)),
            analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-5-imb-initG02"),
        ), 

    "REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-6-imb-initG02": ExperimentConfig(
            name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-6-imb-initG02",
            model=copy.deepcopy(_MODEL_REPLICATION),
            data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
            optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                              weight_decay=0.0, grad_clip=0),
            lr=LRConfig(scheduler="constant", learning_rate=6e-6, min_lr=3e-4,
                        warmup_iters=50),
            train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                              run_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-6-imb-initG02",
                              ckpt_fracs=dict(_CKPT_9)),
            analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-6-imb-initG02"),
        ),

    "REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-7-imb-initG02": ExperimentConfig(
            name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-7-imb-initG02",
            model=copy.deepcopy(_MODEL_REPLICATION),
            data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
            optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                              weight_decay=0.0, grad_clip=0),
            lr=LRConfig(scheduler="constant", learning_rate=6e-7, min_lr=3e-4,
                        warmup_iters=50),
            train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                              run_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-7-imb-initG02",
                              ckpt_fracs=dict(_CKPT_9)),
            analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-7-imb-initG02"),
        ),
# --------------------------Replication--same data construction----------------------------
    "REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-6-imb-initG02-identity": ExperimentConfig(
            name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-6-imb-initG02",
            model=copy.deepcopy(_MODEL_REPLICATION),
            data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
            optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                              weight_decay=0.0, grad_clip=0),
            lr=LRConfig(scheduler="constant", learning_rate=6e-6, min_lr=3e-4,
                        warmup_iters=50),
            train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                              run_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-6-imb-initG02-identity",
                              ckpt_fracs=dict(_CKPT_9)),
            analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr6e-6-imb-initG02-identity"),
        ),

    "REP-mse0-pos0-frz_embd-fullbs-adam-lr1e-5-imb-initG02-identity": ExperimentConfig(
            name="REP-mse0-pos0-frz_embd-fullbs-adam-lr1e-5-imb-initG02",
            model=copy.deepcopy(_MODEL_REPLICATION),
            data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
            optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                              weight_decay=0.0, grad_clip=0),
            lr=LRConfig(scheduler="constant", learning_rate=1e-5, min_lr=3e-4,
                        warmup_iters=50),
            train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                              run_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr1e-5-imb-initG02-identity",
                              ckpt_fracs=dict(_CKPT_9)),
            analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr1e-5-imb-initG02-identity"),
        ),
    
    "REP-mse0-pos0-frz_embd-fullbs-adam-lr8e-6-imb-initG02-identity": ExperimentConfig(
            name="REP-mse0-pos0-frz_embd-fullbs-adam-lr8e-6-imb-initG02",
            model=copy.deepcopy(_MODEL_REPLICATION),
            data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
            optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                              weight_decay=0.0, grad_clip=0),
            lr=LRConfig(scheduler="constant", learning_rate=8e-6, min_lr=3e-4,
                        warmup_iters=50),
            train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                              run_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr8e-6-imb-initG02-identity",
                              ckpt_fracs=dict(_CKPT_9)),
            analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr8e-6-imb-initG02-identity"),
        ),

    "REP-mse0-pos0-frz_embd-fullbs-sgd-imb-initG02-identity": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-sgd-imb-initG02-identity",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=0.05, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-sgd-imb-initG02-identity",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-sgd-imb-initG02-identity"),
    ),

        "REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p07-imb-initG02-identity": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p07-imb-initG02-identity",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=0.07, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p07-imb-initG02-identity",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p07-imb-initG02-identity"),
    ),

        "REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p09-imb-initG02-identity": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p09-imb-initG02-identity",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=0.09, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p09-imb-initG02-identity",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p09-imb-initG02-identity"),
    ),

        "REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p1-imb-initG02-identity": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p1-imb-initG02-identity",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=0.1, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p1-imb-initG02-identity",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p1-imb-initG02-identity"),
    ),

        "REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p15-imb-initG02-identity": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p15-imb-initG02-identity",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=0.15, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p15-imb-initG02-identity",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p15-imb-initG02-identity"),
    ),

        "REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p25-imb-initG02-identity": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p25-imb-initG02-identity",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=0.25, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p25-imb-initG02-identity",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p25-imb-initG02-identity"),
    ),

    
        "REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p5-imb-initG02-identity": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p5-imb-initG02-identity",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=0.5, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p5-imb-initG02-identity",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p5-imb-initG02-identity"),
    ),
    #------------------use mse_rep loss------------------------------
        "REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p005-imb-initG02-identity-nobias": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p005-imb-initG02-identity-nobias",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=0.005, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p005-imb-initG02-identity-nobias",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p005-imb-initG02-identity-nobias"),
    ),

        "REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p001-imb-initG02-identity-nobias": ExperimentConfig(
        name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p001-imb-initG02-identity-nobias",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=0.001, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                          run_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p001-imb-initG02-identity-nobias",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-sgd-lr0p001-imb-initG02-identity-nobias"),
    ),
    
    "REP-mse0-pos0-frz_embd-fullbs-adam-lr1e-5-imb-initG02-identity-nobias": ExperimentConfig(
            name="REP-mse0-pos0-frz_embd-fullbs-adam-lr1e-5-imb-initG02-identity-nobias",
            model=copy.deepcopy(_MODEL_REPLICATION),
            data=DataConfig(dataset="synth_identity_zipf_s1_V10000", batch_size=782),
            optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                              weight_decay=0.0, grad_clip=0),
            lr=LRConfig(scheduler="constant", learning_rate=1e-5, min_lr=3e-5,
                        warmup_iters=50),
            train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_iters=5,
                              run_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr1e-5-imb-initG02-identity-nobias",
                              ckpt_fracs=dict(_CKPT_9)),
            analyze=AnalyzeConfig(files_name="REP-mse0-pos0-frz_embd-fullbs-adam-lr1e-5-imb-initG02-identity-nobias"),
        ),  

#---------------------Replication-Muon---------------------------
    "REP-muon-lr3e-6-G02-mom0": ExperimentConfig(
        name="REP-muon-lr3e-6-G02-mom0",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="muon", weight_decay=0.0, grad_clip=0, muon_momentum=0.0),
        lr=LRConfig(scheduler="constant", learning_rate=3e-6, min_lr=0.075e-4,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_interval=50,
                          run_name="REP-muon-lr3e-6-G02-mom0",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-muon-lr3e-6-G02-mom0"),
    ),

    "REP-muon-lr3e-5-G02-mom0": ExperimentConfig(
        name="REP-muon-lr3e-5-G02-mom0",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="muon", weight_decay=0.0, grad_clip=0, muon_momentum=0.0),
        lr=LRConfig(scheduler="constant", learning_rate=3e-5, min_lr=0.075e-4,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_interval=50,
                          run_name="REP-muon-lr3e-5-G02-mom0",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-muon-lr3e-5-G02-mom0"),
    ),

    "REP-muon-lr6e-5-G02-mom0": ExperimentConfig(
        name="REP-muon-lr6e-5-G02-mom0",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="muon", weight_decay=0.0, grad_clip=0, muon_momentum=0.0),
        lr=LRConfig(scheduler="constant", learning_rate=6e-5, min_lr=0.075e-4,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_interval=50,
                          run_name="REP-muon-lr6e-5-G02-mom0",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-muon-lr6e-5-G02-mom0"),
    ),

    "REP-muon-lr3e-4-G02-mom0": ExperimentConfig(
        name="REP-muon-lr3e-4-G02-mom0",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="muon", weight_decay=0.0, grad_clip=0, muon_momentum=0.0),
        lr=LRConfig(scheduler="constant", learning_rate=3e-4, min_lr=0.075e-4,
                    warmup_iters=50),
        train=TrainConfig(freeze = "tok_emb", max_iters=500, eval_interval=50,
                          run_name="REP-muon-lr3e-4-G02-mom0",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP-muon-lr3e-4-G02-mom0"),
    ),

#-------------------------------Freeze Test Muon/Adam/SGD----------------------
    "REP1-frz_lmhead-muon-lr6e-5-G02": ExperimentConfig(
        name="REP1-frz_lmhead-muon-lr6e-5-G02",
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="muon", weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=6e-5, min_lr=0.075e-4,
                    warmup_iters=50),
        train=TrainConfig(freeze = "lm_head", max_iters=500, eval_iters = 5, eval_interval=50,
                          run_name="REP1-frz_lmhead-muon-lr6e-5-G02",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="REP1-frz_lmhead-muon-lr6e-5-G02"),
    ),

    "REP1-frz_lmhead-adam-lr3e-6-G02": ExperimentConfig(
            name="REP1-frz_lmhead-adam-lr3e-6-G02",
            model=copy.deepcopy(_MODEL_REPLICATION),
            data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
            optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                              weight_decay=0.0, grad_clip=0),
            lr=LRConfig(scheduler="constant", learning_rate=3e-6, min_lr=3e-5,
                        warmup_iters=50),
            train=TrainConfig(freeze = "lm_head", max_iters=500, eval_iters=5, eval_interval=50,
                              run_name="REP1-frz_lmhead-adam-lr3e-6-G02",
                              ckpt_fracs=dict(_CKPT_9)),
            analyze=AnalyzeConfig(files_name="REP1-frz_lmhead-adam-lr3e-6-G02"),
        ),

    "REP1-frz_lmhead-sgd-lr0p01": ExperimentConfig(
    name="REP1-frz_lmhead-sgd-lr0p01",
    model=copy.deepcopy(_MODEL_REPLICATION),
    data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
    optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
    lr=LRConfig(scheduler="constant", learning_rate=0.01, min_lr=0.0025,
                    warmup_iters=50),
    train=TrainConfig(freeze = "lm_head", max_iters=500, eval_iters=5, eval_interval=50,    
                          run_name="REP1-frz_lmhead-sgd-lr0p01",
                          ckpt_fracs=dict(_CKPT_9)),
    analyze=AnalyzeConfig(files_name="REP1-frz_lmhead-sgd-lr0p01"),
    ),

    "REP1-frz_lmhead-sgd-lr0p005": ExperimentConfig(
    name="REP1-frz_lmhead-sgd-lr0p005",
    model=copy.deepcopy(_MODEL_REPLICATION),
    data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
    optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
    lr=LRConfig(scheduler="constant", learning_rate=0.005, min_lr=0.0025,
                    warmup_iters=50),
    train=TrainConfig(freeze = "lm_head", max_iters=500, eval_iters=5, eval_interval=50,    
                          run_name="REP1-frz_lmhead-sgd-lr0p005",
                          ckpt_fracs=dict(_CKPT_9)),
    analyze=AnalyzeConfig(files_name="REP1-frz_lmhead-sgd-lr0p005"),
    ),

#---------------REP1 LR sweep: freeze lm_head, V=10000 d=10000---------------
# SGD: fine-tune around the known best lr=0.01 (stable boundary ~0.011)

# SGD 0.007
"REP1-frz_lmhead-sgd-lr0p007-G02": ExperimentConfig(
    name="REP1-frz_lmhead-sgd-lr0p007-G02",
    model=copy.deepcopy(_MODEL_REPLICATION),
    data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
    optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
    lr=LRConfig(scheduler="constant", learning_rate=0.007, min_lr=0.0025,
                warmup_iters=50),
    train=TrainConfig(freeze="lm_head", max_iters=500, eval_iters=5, eval_interval=50,
                      run_name="REP1-frz_lmhead-sgd-lr0p007-G02",
                      ckpt_fracs=dict(_CKPT_9)),
    analyze=AnalyzeConfig(files_name="REP1-frz_lmhead-sgd-lr0p007-G02"),
),

# SGD 0.013
"REP1-frz_lmhead-sgd-lr0p013-G02": ExperimentConfig(
    name="REP1-frz_lmhead-sgd-lr0p013-G02",
    model=copy.deepcopy(_MODEL_REPLICATION),
    data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
    optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
    lr=LRConfig(scheduler="constant", learning_rate=0.013, min_lr=0.0025,
                warmup_iters=50),
    train=TrainConfig(freeze="lm_head", max_iters=500, eval_iters=5, eval_interval=50,
                      run_name="REP1-frz_lmhead-sgd-lr0p013-G02",
                      ckpt_fracs=dict(_CKPT_9)),
    analyze=AnalyzeConfig(files_name="REP1-frz_lmhead-sgd-lr0p013-G02"),
),

# SGD 0.017
"REP1-frz_lmhead-sgd-lr0p017-G02": ExperimentConfig(
    name="REP1-frz_lmhead-sgd-lr0p017-G02",
    model=copy.deepcopy(_MODEL_REPLICATION),
    data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
    optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
    lr=LRConfig(scheduler="constant", learning_rate=0.017, min_lr=0.0025,
                warmup_iters=50),
    train=TrainConfig(freeze="lm_head", max_iters=500, eval_iters=5, eval_interval=50,
                      run_name="REP1-frz_lmhead-sgd-lr0p017-G02",
                      ckpt_fracs=dict(_CKPT_9)),
    analyze=AnalyzeConfig(files_name="REP1-frz_lmhead-sgd-lr0p017-G02"),
),

#------------------------Freeze Test-----------------------------------
# linear model (only embedding + lm_head) + mes loss + imbalance data (initialize = 1)
    # ---------freeze embedding + SGD---------------------------
    "mse0-frozen_embd-sgd-imbalance": ExperimentConfig(
        name="mse0-frozen_embd-sgd-imbalance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.1, min_lr=0.005,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=8000, run_name="mse0_frozen_embd-sgd-imbalance",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0_frozen_embd-sgd-imbalance"),
    ),
    # ---------freeze embedding + AdamW--------------------------
    "mse0-frozen_embd-adamw-imbalance": ExperimentConfig(
        name="mse0-frozen_embd-adamw-imbalance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-4, min_lr=0.075e-4,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=8000, run_name="mse0_frozen_embd-adamw-imbalance",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0_frozen_embd-adamw-imbalance"),
    ),

    # ---------freeze lm_head + SGD lr0.1------------------------------
    "mse0-frozen_lmhead-sgd-lr0p1-imbalance": ExperimentConfig(
        name="mse0-frozen_lmhead-sgd-lr0p1-imbalance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.1, min_lr=0.005,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=8000, run_name="mse0_frozen_lmhead-sgd-lr0p1-imbalance",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0_frozen_lmhead-sgd-lr0p1-imbalance"),
    ),

    # ---------freeze lm_head + SGD lr = 0.05------------------------------
    "mse0-frozen_lmhead-sgd-lr0p05-imbalance": ExperimentConfig(
        name="mse0-frozen_lmhead-sgd-lr0p05-imbalance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.05, min_lr=0.0025,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=8000, run_name="mse0_frozen_lmhead-sgd-lr0p05-imbalance",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0_frozen_lmhead-sgd-lr0p05-imbalance"),
    ),

    # ---------freeze lm_head + AdamW----------------------------
    "mse0-frozen_lmhead-adamw-imbalance": ExperimentConfig(
        name="mse0-frozen_lmhead-adamw-imbalance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-4, min_lr=0.075e-4,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=8000, run_name="mse0_frozen_lmhead-adamw-imbalance",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0_frozen_lmhead-adamw-imbalance"),
    ),

    # ---------freeze embedding + AdamW lr = 1.5e-3--------------------------
    "mse0-frozen_embd-adamw-lr1p5e-3-imbalance": ExperimentConfig(
        name="mse0-frozen_embd-adamw-lr1p5e-3-imbalance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-3, min_lr=0.075e-3,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=8000, run_name="mse0_frozen_embd-adamw-lr1p5e-3-imbalance",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0_frozen_embd-adamw-lr1p5e-3-imbalance"),
    ),

    # ---------freeze lm_head + AdamW lr = 1.5e-3----------------------------
    "mse0-frozen_lmhead-adamw-lr1p5e-3-imbalance": ExperimentConfig(
        name="mse0-frozen_lmhead-adamw-lr1p5e-3-imbalance",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-3, min_lr=0.075e-3,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=8000, run_name="mse0_frozen_lmhead-adamw-lr1p5e-3-imbalance",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0_frozen_lmhead-adamw-lr1p5e-3-imbalance"),
    ),




}

# Adam (β1=0, β2=0.999): geometric sweep around 3e-6
for _lr, _tag in [(1e-6, "1e-6"), (2e-6, "2e-6"), (5e-6, "5e-6"),
                  (1e-5, "1e-5"), (2e-5, "2e-5"), (5e-5, "5e-5")]:
    _nm = f"REP1-frz_lmhead-adam-lr{_tag}-G02"
    EXPERIMENTS[_nm] = ExperimentConfig(
        name=_nm,
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                          weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=_lr, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze="lm_head", max_iters=500, eval_iters=5,
                          eval_interval=50, run_name=_nm, ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name=_nm),
    )

# Muon: geometric sweep around 6e-5
for _lr, _tag in [(2e-5, "2e-5"), (4e-5, "4e-5"), (1e-4, "1e-4"),
                  (2e-4, "2e-4"), (4e-4, "4e-4"), (6e-5,"6e-5")]:
    _nm = f"REP1-frz_lmhead-muon-lr{_tag}-G02-mom0"
    EXPERIMENTS[_nm] = ExperimentConfig(
        name=_nm,
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="muon", weight_decay=0.0, grad_clip=0, muon_momentum = 0.0),
        lr=LRConfig(scheduler="constant", learning_rate=_lr, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze="lm_head", max_iters=500, eval_iters=5,
                          eval_interval=50, run_name=_nm, ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name=_nm),
    )








# ---- REP bigram + bias + mse: SGD lr sweep -----------------------------
# The frozen initG02 embedding gives a GN Hessian (per lm_head row, identical
# for every class) H = (2/V) sum_x rho_x [emb_x;1][emb_x;1]^T with lambda_max
# = 7.88e-3 (power iteration on the exact closed form), so full-batch GD is
# stable for lr < 2/lambda_max ~= 254.  The earlier lr=0.05 run sat ~3 orders
# of magnitude below that and barely moved; this grid walks up to the bound.
for _lr in [1.0, 5.0, 20.0, 60.0, 120.0, 200.0, 240.0]:
    _tag = str(_lr).replace(".0", "").replace(".", "p")
    _nm = f"REP-mse0-pos0-frz_embd-fullbs-sgd-lr{_tag}-imb-initG02"
    EXPERIMENTS[_nm] = ExperimentConfig(
        name=_nm,
        model=copy.deepcopy(_MODEL_REPLICATION_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=_lr, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze="tok_emb", max_iters=500, eval_iters=5,
                          run_name=_nm, ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name=_nm),
    )

# ---- REP bigram + bias + mse: Adam lr sweep ----------------------------
# The earlier Adam grid jumped a full decade from 6e-6 (clean, but the
# per-freq-group decay is still strongly stratified -- possibly just
# undertrained) to 6e-5 (12 loss spikes).  This grid fills the gap with
# ~1.5x geometric spacing to find the largest spike-free lr.
for _lr, _tag in [(8e-6, "8e-6"), (1.2e-5, "1.2e-5"), (1.8e-5, "1.8e-5"),
                  (2.7e-5, "2.7e-5"), (4e-5, "4e-5")]:
    _nm = f"REP-mse0-pos0-frz_embd-fullbs-adam-lr{_tag}-imb-initG02"
    EXPERIMENTS[_nm] = ExperimentConfig(
        name=_nm,
        model=copy.deepcopy(_MODEL_REPLICATION_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                          weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=_lr, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze="tok_emb", max_iters=500, eval_iters=5,
                          run_name=_nm, ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name=_nm),
    )

# ---- REP bigram + NO bias + mse: SGD & Adam lr sweeps ------------------
# Same cell as the two sweeps above (frozen initG02 embedding, bigram data,
# mse loss, full batch, 500 iters) but with the lm_head bias removed, to test
# whether the bias was what kept the per-frequency-group decay stratified.
# The exact GN lambda_max is 7.8948e-3 without the bias vs 7.9151e-3 with it
# (power iteration on the closed form), i.e. the stability bound is unchanged
# at lr < 2/lambda_max ~= 253, so both grids are reused as-is.
# NOTE: these runs require lm_head bias=False in vanilla_model.py.
for _lr in [1.0, 5.0, 20.0, 60.0, 120.0, 200.0, 240.0]:
    _tag = str(_lr).replace(".0", "").replace(".", "p")
    _nm = f"REP-mse0-pos0-frz_embd-fullbs-sgd-lr{_tag}-imb-initG02-nobias"
    EXPERIMENTS[_nm] = ExperimentConfig(
        name=_nm,
        model=copy.deepcopy(_MODEL_REPLICATION_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=_lr, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze="tok_emb", max_iters=500, eval_iters=5,
                          run_name=_nm, ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name=_nm),
    )

for _lr, _tag in [(6e-7, "6e-7"), (2e-6, "2e-6"), (6e-6, "6e-6"),
                  (1.2e-5, "1.2e-5"), (2.7e-5, "2.7e-5"), (6e-5, "6e-5"),
                  (2e-4, "2e-4")]:
    _nm = f"REP-mse0-pos0-frz_embd-fullbs-adam-lr{_tag}-imb-initG02-nobias"
    EXPERIMENTS[_nm] = ExperimentConfig(
        name=_nm,
        model=copy.deepcopy(_MODEL_REPLICATION_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                          weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=_lr, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze="tok_emb", max_iters=500, eval_iters=5,
                          run_name=_nm, ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name=_nm),
    )

# ---- REP bigram + NO bias + mse_rep: the eps test ----------------------
# The nobias mse sweep (above) still stratifies under Adam, ruling the bias
# out.  At W=0 in the mse convention 98.8% of lm_head gradient coords are
# BELOW Adam's eps=1e-8 (median |g| = 3.6e-10), so Adam's denominator is eps
# for almost every coordinate and it degenerates into plain GD with effective
# lr = lr/eps -- which is why it fans out like SGD.  The mse_rep convention
# (0.5*sum = V/2 = 5000x) lifts the median |g| to 1.8e-6 >> eps: if eps is the
# cause, the same Adam here should equalize the groups (as replication/ does
# with the same 0.5*sum loss).  SGD is lr-rescaled by 1/5000 (lambda_max =
# 5000 * 7.89e-3 = 39.5, GD stable lr < 0.0507).
for _lr, _tag in [(0.048, "0p048"), (0.07, "0p07"), (0.09, "0p09")]:
    _nm = f"REP-mserep-pos0-frz_embd-fullbs-sgd-lr{_tag}-imb-initG02-nobias"
    EXPERIMENTS[_nm] = ExperimentConfig(
        name=_nm,
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="sgd", momentum=0.0, weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=_lr, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze="tok_emb", max_iters=500, eval_iters=5,
                          run_name=_nm, ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name=_nm),
    )

for _lr, _tag in [(3e-6, "3e-6"), (9e-6, "9e-6"), (2e-6, "2e-6")]:
    _nm = f"REP-mserep-pos0-frz_embd-fullbs-adam-lr{_tag}-imb-initG02-nobias"
    EXPERIMENTS[_nm] = ExperimentConfig(
        name=_nm,
        model=copy.deepcopy(_MODEL_REPLICATION),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V10000", batch_size=782),
        optim=OptimConfig(name="adam", betas=(0.0, 0.999), eps=1e-8,
                          weight_decay=0.0, grad_clip=0),
        lr=LRConfig(scheduler="constant", learning_rate=_lr, min_lr=0.0025,
                    warmup_iters=50),
        train=TrainConfig(freeze="tok_emb", max_iters=500, eval_iters=5,
                          run_name=_nm, ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name=_nm),
    )



# ---- V1024 initG frozen_embd SGD lr sweep -----------------------------------
# Tune lr for mse0-pos0-frozen_embd-sgd-imb-initG-iter1500 (V1024, emb and W
# both N(0,0.02), bias on, freeze tok_emb, mom=0.9, wd=0.1, cosine wu200).
# Exact GN of the frozen-emb problem: lambda_max = 1.96e-3, so total curvature
# is DOMINATED by the coupled wd=0.1; heavy-ball stability lr < 2(1+b)/(lam+wd)
# ~= 37.  The original lr=0.1 is ~370x below that; the grid walks up to it.
# min_lr = lr/20, matching the original preset's ratio.
_MODEL_V1024_MSE_POS0_G002 = ModelConfig(
    vocab_size=1024, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=0, block_size=128, loss_type="mse",
    use_pos_enc=False, 
)

for _lr, _tag in [(0.3, "0p3"), (1.0, "1"), (3.0, "3"), (10.0, "10"),
                  (30.0, "30")]:
    _nm = f"mse0-pos0-frozen_embd-sgd-lr{_tag}-imb-initG-iter1500"
    EXPERIMENTS[_nm] = ExperimentConfig(
        name=_nm,
        model=copy.deepcopy(_MODEL_V1024_MSE_POS0_G002),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1,
                          grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=_lr, min_lr=_lr / 20,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=1500, run_name=_nm,
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name=_nm),
    )



# the preset load() falls back to when no name is given
DEFAULT = ""


def get(name):
    if name not in EXPERIMENTS:
        raise KeyError(
            f"unknown experiment {name!r}; known: {sorted(EXPERIMENTS)}"
        )
    return copy.deepcopy(EXPERIMENTS[name])
