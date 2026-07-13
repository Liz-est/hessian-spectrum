# Balanced data: uniform token frequency (pi = uniform).
# All tokens equally frequent -> no class imbalance in the softmax head.
# Use as the "balance" baseline against zipf_imbalanced.py.

vocab_size = 2048
n_train_tokens = 10_000_000
n_val_tokens = 100_000

freq = "uniform"            # <-- balanced

predictability = 0.8        # moderately learnable next-token structure
bandwidth_frac = 0.02

label_mode = "shift"        # default: y = input shifted by 1
shift = 1

seed = 1337
out_dir = "data/synth_uniform_balanced"
