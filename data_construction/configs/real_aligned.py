# Real-corpus-aligned frequency: pi matches an empirical unigram distribution.
#
# Prepare `real_counts_path` as a .npy 1-D array of length vocab_size holding
# per-token counts (or frequencies) from a tokenized real corpus, e.g.:
#
#     import numpy as np, collections
#     ids = np.fromfile("your_tokenized_corpus.bin", dtype=np.uint16)
#     counts = np.bincount(ids, minlength=vocab_size).astype(np.float64)
#     np.save("data/real_unigram_counts.npy", counts)
#
# The sequence structure is still the synthetic bigram chain; only the marginal
# token frequency pi is aligned to reality.  To also flatten it into a matched
# "balanced" version, set freq="uniform" with the same vocab_size.

vocab_size = 2048
n_train_tokens = 10_000_000
n_val_tokens = 100_000

freq = "real"
real_counts_path = "data/real_unigram_counts.npy"   # <-- provide this

predictability = 0.8
bandwidth_frac = 0.02

label_mode = "shift"
shift = 1

seed = 1337
out_dir = "data/synth_real_aligned"
