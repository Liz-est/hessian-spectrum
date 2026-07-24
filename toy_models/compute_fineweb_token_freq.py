"""
Unigram token counts for the fineweb10B train split.

Streams every data/fineweb10B/fineweb_train_*.bin shard (modded-nanoGPT format:
256 x int32 header, then a uint16 token stream) through a chunked np.bincount
and writes data/fineweb10B/token_counts.npy (int64, length vocab_size). The
analyzer's token_select="freq" mode reads this file to pick the top-N most
frequent token ids for the embedding / lm_head per-token Hessian blocks.

Idempotent: exits immediately if the output already exists (delete it to
rebuild). Memory use is one chunk of tokens at a time (~20 MB), safe under the
dev box's 8 GB cgroup.

    python3 compute_fineweb_token_freq.py
"""

import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data", "fineweb10B")
OUT_PATH = os.path.join(DATA_DIR, "token_counts.npy")

VOCAB_SIZE = 50304          # GPT-2 BPE padded, matches the fineweb presets
HEADER_BYTES = 256 * 4
CHUNK = 10_000_000


def main():
    if os.path.exists(OUT_PATH):
        counts = np.load(OUT_PATH)
        print(f"{OUT_PATH} already exists ({counts.sum():,} tokens); delete to rebuild.")
        return

    paths = sorted(glob.glob(os.path.join(DATA_DIR, "fineweb_train_*.bin")))
    assert paths, f"no shards found in {DATA_DIR}"
    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    t0 = time.time()
    for p in paths:
        header = np.fromfile(p, dtype=np.int32, count=256)
        assert header[0] == 20240520, f"bad magic in {p}: {header[0]}"
        ntok = int(header[2])
        toks = np.memmap(p, dtype=np.uint16, mode="r", offset=HEADER_BYTES)[:ntok]
        for s in range(0, ntok, CHUNK):
            counts += np.bincount(toks[s:s + CHUNK], minlength=VOCAB_SIZE)
        print(f"{os.path.basename(p)}: {ntok:,} tokens "
              f"(total {counts.sum():,}, {time.time() - t0:.0f}s)", flush=True)

    np.save(OUT_PATH, counts)
    top = np.argsort(-counts, kind="stable")[:10]
    print(f"wrote {OUT_PATH}: {counts.sum():,} tokens, "
          f"{int((counts > 0).sum())}/{VOCAB_SIZE} ids seen")
    print("top-10 ids:", top.tolist())
    print("top-10 counts:", counts[top].tolist())


if __name__ == "__main__":
    main()
