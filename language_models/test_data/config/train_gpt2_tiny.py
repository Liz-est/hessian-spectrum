# Tiny transformer config for the Adam-vs-SGD / last-vs-hidden-layer Hessian study
# on the synth_uniform_balanced (order-1 Markov) dataset.
#
# The data only needs the previous token to predict the next, so a short context
# and a small model are plenty. This trains in minutes on a single A100 while
# still keeping enough depth (4 blocks) to compare last-layer vs hidden-layer
# Hessian heterogeneity.

# ---- data batching --------------------------------------------------------
batch_size = 64                 # single-batch fits easily; feeds the GPU
block_size = 128                # order-1 Markov: long context is wasted
gradient_accumulation_steps = 1 # no need to simulate a huge batch here

# ---- model ----------------------------------------------------------------
n_layer = 4                     # keep multiple layers for last-vs-hidden contrast
n_head  = 4
n_embd  = 256
dropout = 0.0
bias    = False

# ---- weight tying ---------------------------------------------------------
# Disabled so lm_head is an INDEPENDENT tensor from the input embedding.
# This lets the Hessian of the "last layer" be attributed cleanly.
tie_weights = False

# ---- schedule -------------------------------------------------------------
max_iters       = 8000
lr_decay_iters  = 8000
warmup_iters    = 200

# ---- eval / logging / ckpt ------------------------------------------------
eval_interval = 200
eval_iters    = 100
log_interval  = 20
ckpt_interval = 1000

# ---- optimizer ------------------------------------------------------------
# use_sgd is overridden per run script (Adam vs SGD). Default here = Adam.
use_sgd       = False
learning_rate = 6e-4            # good for AdamW; SGD run script overrides this much higher
weight_decay  = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 3e-5

dtype = 'float32'

init_from = 'scratch'
load_iter = 0

# ---- layers to sample for the layer-by-layer Hessian spectrum -------------
# With tying disabled, wte (input embedding) and lm_head (output / last layer)
# are separate parameters and are listed separately so their spectra can be
# compared against the hidden blocks.
sample_layer = [
    'transformer.wte.weight',            # input embedding
    'transformer.h.0.attn.wq.weight',
    'transformer.h.0.attn.wk.weight',
    'transformer.h.0.attn.wv.weight',
    'transformer.h.0.attn.wo.weight',
    'transformer.h.0.mlp.c_fc.weight',
    'transformer.h.0.mlp.c_proj.weight',
    'transformer.h.1.attn.wq.weight',
    'transformer.h.1.attn.wk.weight',
    'transformer.h.1.attn.wv.weight',
    'transformer.h.1.attn.wo.weight',
    'transformer.h.1.mlp.c_fc.weight',
    'transformer.h.1.mlp.c_proj.weight',
    'transformer.h.2.attn.wq.weight',
    'transformer.h.2.attn.wk.weight',
    'transformer.h.2.attn.wv.weight',
    'transformer.h.2.attn.wo.weight',
    'transformer.h.2.mlp.c_fc.weight',
    'transformer.h.2.mlp.c_proj.weight',
    'transformer.h.3.attn.wq.weight',
    'transformer.h.3.attn.wk.weight',
    'transformer.h.3.attn.wv.weight',
    'transformer.h.3.attn.wo.weight',
    'transformer.h.3.mlp.c_fc.weight',
    'transformer.h.3.mlp.c_proj.weight',
    'lm_head.weight',                    # output / LAST layer (independent once untied)
]

comment  = 'tiny_hessian'      # used by tensorboard + hessian file dir
save_dir = 'log_gpt2/' + comment
out_dir  = 'out-gpt2/' + comment
