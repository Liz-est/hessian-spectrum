"""
Archived experiment presets.

These historical configurations are intentionally NOT registered by
config.load(). The active registry lives in presets.py and contains only the
current full-batch comparison.

Import this module explicitly only when inspecting or recovering an old
experiment. Moving a configuration back into presets.py is the deliberate step
that makes it active again.
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
# (same shape, n_layer=1).
_MODEL_fw10B_l1 = ModelConfig(
    vocab_size=50304, n_embd=192, n_head=6, head_dim=32,
    n_ffn=1024, n_layer=1, block_size=1024,
)

_CKPT_9 = {"init": 0.0, "p10": 0.10, "p25": 0.25, "p40": 0.40, "p50": 0.50,
           "p60": 0.60, "p75": 0.75, "p85": 0.85, "p100": 1.0}


LEGACY_EXPERIMENTS = {
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
#   GPT-2 BPE (vocab 50304), 1024-token context, AdamW. Training budget: 20k iters,
#   bs=32, warmup=400, lr 6e-4 -> 6e-5. Data lives in <repo-root>/data/fineweb10B/
#   as modded-nanoGPT
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
    "layer5-mse-imbalance-s1-muon-gradclip0": ExperimentConfig(
        name="layer5-mse-imbalance-s1-muon-gradclip0",
        model=copy.deepcopy(_MODEL_l5_MSE),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="muon", weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-4, min_lr=0.075e-4,
                    warmup_iters=200),
        train=TrainConfig(max_iters=8000, run_name="layer5-mse-imbalance-s1-muon-gradclip0",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="layer5-mse-imbalance-s1-muon-gradclip0"),
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
    "mse0-pos0-frozen_embd-sgd-lr0p1-imb-init1G-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_embd-sgd-lr0p1-imb-init1G-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.1, min_lr=0.005,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=1500, run_name="mse0-pos0-frozen_embd-sgd-lr0p1-imb-init1G-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_embd-sgd-lr0p1-imb-init1G-iter1500"),
    ),
    # ---------freeze embedding + AdamW lr = 1.5e-3--------------------------
    "mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-init1G-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-init1G-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-3, min_lr=0.075e-3,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=1500, run_name="mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-init1G-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-init1G-iter1500"),
    ),

    # ---------freeze lm_head + SGD lr = 0.05------------------------------
    "mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="sgd", momentum=0.9, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=0.05, min_lr=0.0025,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=1500, run_name="mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G-iter1500"),
    ),

    # ---------freeze lm_head + AdamW lr = 1.5e-3----------------------------
    "mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=1.5e-3, min_lr=0.075e-3,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=1500, run_name="mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G-iter1500"),
    ),

        # ---------freeze lm_head + AdamW lr = 6e-3----------------------------
    "mse0-pos0-frozen_lmhead-adamw-lr6e-3-imb-init1G-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_lmhead-adamw-lr6e-3-imb-init1G-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-3, min_lr=3e-4,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=1500, run_name="mse0-pos0-frozen_lmhead-adamw-lr6e-3-imb-init1G-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_lmhead-adamw-lr6e-3-imb-init1G-iter1500"),
    ),

            # ---------freeze embedding + AdamW lr = 6e-3----------------------------
    "mse0-pos0-frozen_embd-adamw-lr6e-3-imb-init1G-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_embd-adamw-lr6e-3-imb-init1G-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=6e-3, min_lr=3e-4,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=1500, run_name="mse0-pos0-frozen_embd-adamw-lr6e-3-imb-init1G-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_embd-adamw-lr6e-3-imb-init1G-iter1500"),
    ),

            # ---------freeze lm_head + AdamW lr = 9e-3----------------------------
    "mse0-pos0-frozen_lmhead-adamw-lr9e-3-imb-init1G-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_lmhead-adamw-lr9e-3-imb-init1G-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=9e-3, min_lr=4.5e-4,
                    warmup_iters=200),
        train=TrainConfig(freeze="lm_head", max_iters=1500, run_name="mse0-pos0-frozen_lmhead-adamw-lr9e-3-imb-init1G-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_lmhead-adamw-lr9e-3-imb-init1G-iter1500"),
    ),

            # ---------freeze embedding + AdamW lr = 9e-3----------------------------
    "mse0-pos0-frozen_embd-adamw-lr9e-3-imb-init1G-iter1500": ExperimentConfig(
        name="mse0-pos0-frozen_embd-adamw-lr9e-3-imb-init1G-iter1500",
        model=copy.deepcopy(_MODEL_EMBED_HEAD_MSE_POS0),
        data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
        optim=OptimConfig(name="adamw", betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, grad_clip=0),
        lr=LRConfig(scheduler="cosine", learning_rate=9e-3, min_lr=4.5e-4,
                    warmup_iters=200),
        train=TrainConfig(freeze="tok_emb", max_iters=1500, run_name="mse0-pos0-frozen_embd-adamw-lr9e-3-imb-init1G-iter1500",
                          ckpt_fracs=dict(_CKPT_9)),
        analyze=AnalyzeConfig(files_name="mse0-pos0-frozen_embd-adamw-lr9e-3-imb-init1G-iter1500"),
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

# the preset load() falls back to when no name is given
DEFAULT = ""


def get(name):
    if name not in LEGACY_EXPERIMENTS:
        raise KeyError(
            f"unknown legacy experiment {name!r}; "
            f"known: {sorted(LEGACY_EXPERIMENTS)}"
        )
    return copy.deepcopy(LEGACY_EXPERIMENTS[name])
