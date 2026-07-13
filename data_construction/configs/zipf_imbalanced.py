# Imbalanced data: Zipf token frequency (pi ~ 1 / rank^s).
# A few tokens are very frequent, a long tail is rare -> strong class imbalance
# in the softmax head.  Same predictability as uniform_balanced.py, so the ONLY
# thing that differs is pi -> clean balance-vs-imbalance comparison for Hessian.

vocab_size = 2048
n_train_tokens = 10_000_000
n_val_tokens = 100_000

freq = "zipf"              # <-- imbalanced
zipf_s = 1.0               # 0 -> uniform, 1 -> classic Zipf, >1 -> heavier tail

predictability = 0.8       # SAME as the balanced config (difficulty held fixed)
bandwidth_frac = 0.02

label_mode = "shift"
shift = 1

seed = 1337
out_dir = "data/synth_zipf_imbalanced"
