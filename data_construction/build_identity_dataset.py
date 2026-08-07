"""
build_identity_dataset.py
=========================
Generate the SIMPLEST possible synthetic dataset for Hessian analysis:

    y_t == x_t          (target is literally the input token, "identity" task)
    x_t  ~ i.i.d. pi    with pi a standard Zipf law  pi[r] ~ 1 / rank^s

There is no sequential structure at all: every token is drawn independently
from pi, and the label is a copy of the input.  The Bayes-optimal CE loss is
exactly 0, so any residual loss is pure optimisation / capacity error, and the
class imbalance seen by the softmax head is set solely by the Zipf exponent.

Output is the same *dual-stream* on-disk format as build_dataset.py, so the
trainer / Hessian estimator / eval scripts work unchanged:

    <out_dir>/
        train_x.bin   uint16  input  token ids
        train_y.bin   uint16  target token ids  (identical to train_x)
        val_x.bin     uint16
        val_y.bin     uint16
        meta.pkl      dict with vocab_size, pi, label_mode="identity", ...

Usage
-----
    python build_identity_dataset.py configs/identity_zipf.py
    python build_identity_dataset.py configs/identity_zipf.py zipf_s=1.5 out_dir=data/my_ds

Any `key=value` on the command line overrides the config (same convention as
build_dataset.py).
"""

import os
import ast
import sys
import pickle

import numpy as np

import transition as T


# --------------------------------------------------------------------------- #
# Default configuration (overridden by a config file and/or key=value args)    #
# --------------------------------------------------------------------------- #
CONFIG = dict(
    vocab_size=1024,
    n_train_tokens=10_000_000,     # total tokens in the training stream
    n_val_tokens=100_000,          # total tokens in the validation stream

    # token-frequency distribution of x (and hence of y, since y == x)
    zipf_s=1.0,                    # 0 -> uniform, 1 -> classic Zipf, >1 heavier

    batch_tokens=4_194_304,        # tokens sampled per write batch (~8 MB uint16)
    seed=1337,
    out_dir="data/synth_identity_zipf_s1_V1024",
)


# --------------------------------------------------------------------------- #
# Config loading (config file first, then key=value overrides)                 #
# --------------------------------------------------------------------------- #
def load_config(argv):
    cfg = dict(CONFIG)
    for arg in argv:
        if arg.endswith(".py"):
            ns = {}
            with open(arg) as f:
                exec(f.read(), {}, ns)
            for k, v in ns.items():
                if not k.startswith("_"):
                    cfg[k] = v
            print(f"[config] loaded {arg}")
        elif "=" in arg:
            k, v = arg.split("=", 1)
            if k not in cfg:
                print(f"[config] warning: unknown key '{k}' (added anyway)")
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass
            cfg[k] = v
            print(f"[config] override {k} = {v!r}")
        else:
            raise ValueError(f"unrecognised argument: {arg}")
    return cfg


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def build(cfg):
    rng = np.random.default_rng(cfg["seed"])
    V = cfg["vocab_size"]
    if V > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"vocab_size={V} does not fit in uint16")

    pi = T.make_pi(V, kind="zipf", zipf_s=cfg["zipf_s"])
    ent = -(pi * np.log(pi)).sum()
    print(f"[build] vocab={V}  zipf_s={cfg['zipf_s']}  H(pi)={ent:.3f} nats "
          f"(max possible={np.log(V):.3f})")
    print(f"[build] identity task: y == x, Bayes-optimal CE loss = 0")

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    write_stream("train", cfg["n_train_tokens"], V, pi, cfg, rng, out_dir)
    write_stream("val", cfg["n_val_tokens"], V, pi, cfg, rng, out_dir)

    meta = dict(
        vocab_size=V,
        pi=pi,                 # marginal of x == marginal of y
        pi_y=pi,
        P=None,                # no sequential structure: x is i.i.d.
        K=None,                # y|x is deterministic identity, no kernel needed
        row_entropy_mean=0.0,  # H(y|x) = 0 exactly
        label_mode="identity",
        dual_stream=True,
        n_train_tokens=cfg["n_train_tokens"],
        n_val_tokens=cfg["n_val_tokens"],
        config=cfg,
        seed=cfg["seed"],
    )
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"[write] meta.pkl  -> {out_dir}")
    print("[done]")


def write_stream(prefix, n_tokens, V, pi, cfg, rng, out_dir):
    """Write <prefix>_x.bin and an identical <prefix>_y.bin, streaming.

    x is drawn i.i.d. from pi in `batch_tokens`-sized chunks and appended to
    both files, so peak memory is O(batch_tokens) regardless of n_tokens.
    Both streams are exactly n_tokens long (no shift, no carry).
    """
    batch = max(1, int(cfg["batch_tokens"]))
    path_x = os.path.join(out_dir, f"{prefix}_x.bin")
    path_y = os.path.join(out_dir, f"{prefix}_y.bin")
    seen = 0
    with open(path_x, "wb") as fx, open(path_y, "wb") as fy:
        while seen < n_tokens:
            b = min(batch, n_tokens - seen)
            seg = rng.choice(V, size=b, p=pi).astype(np.uint16)
            seg.tofile(fx)
            seg.tofile(fy)          # y == x, byte-for-byte
            seen += b
            print(f"[write] {prefix}: {seen:,} / {n_tokens:,} tokens sampled",
                  flush=True)
    print(f"[write] {prefix}_x.bin / {prefix}_y.bin  ({n_tokens:,} tokens each)")
    return n_tokens


if __name__ == "__main__":
    cfg = load_config(sys.argv[1:])
    build(cfg)
