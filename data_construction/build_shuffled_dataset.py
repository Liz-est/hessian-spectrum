"""
build_shuffled_dataset.py
=========================
Generate a synthetic dataset that controls ONLY the two marginal token
distributions -- input pi_x and output pi_y -- and deliberately leaves the
conditional distribution pi_{j|i} uncontrolled.  This is a counterpart to
build_dataset.py, which shapes the conditional structure through a Markov
chain (label_mode shift/independent/coupled); here there is no chain at all.

Construction
------------
1.  pi_x = Zipf(alpha_x), pi_y = Zipf(alpha_y)  (alpha = 0 -> uniform).
2.  Exact-count multisets: token i appears round(N * pi_i) times (largest-
    remainder apportionment), so the EMPIRICAL marginal equals the target
    exactly, not just in expectation.
3.  Independently shuffle the x multiset and the y multiset, then pair them
    by position: (x_t, y_t).  The x/y correspondence is pure random matching,
    so the conditional distribution is whatever the shuffle produces
    (x and y independent in the population sense), and there is no temporal
    correlation along t either.

On-disk format is identical to build_dataset.py (dual-stream), so all
downstream trainers / Hessian estimators read it with zero changes:

    <out_dir>/
        train_x.bin   uint16
        train_y.bin   uint16
        val_x.bin     uint16
        val_y.bin     uint16
        meta.pkl      dict (P=None, K=None, label_mode="shuffled_marginals")

Train/val semantics (generation is decoupled from training)
------------------------------------------------------------
Both splits are always written; how they are used is the trainer's decision:

* minibatch experiments: sample minibatches from train, evaluate on val.
* full-batch experiments: generate a dataset whose train split IS the full
  batch (the default n_train_tokens=2**17 is sized for this), load the
  whole train split as one batch, and run eval / Hessian analysis on train
  as well -- L_train is the exact objective the optimizer descends, so its
  gradient/Hessian are the training dynamics.  val is an optional
  generalization check, unused by training.

Usage
-----
    # full-batch dataset (defaults: 2^17 train pairs + 2^14 val pairs)
    python build_shuffled_dataset.py

    # minibatch-scale dataset
    python build_shuffled_dataset.py n_train_tokens=1000000 n_val_tokens=50000 out_dir=data/my_big

    # other knobs
    python build_shuffled_dataset.py vocab_size=2048 alpha_x=1.0 alpha_y=0.5
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
    # --- vocabulary & size ---------------------------------------------------
    vocab_size=1024,
    n_train_tokens=2**17,          # 131,072 pairs in the training stream;
                                   # default sized so the whole train split can
                                   # serve as ONE full batch
    n_val_tokens=2**14,            # 16,384 pairs in the validation stream
                                   # (optional generalization check)

    # --- marginals: the ONLY thing this construction controls ----------------
    alpha_x=1.0,                   # zipf exponent for the input marginal pi_x
    alpha_y=0.0,                   # zipf exponent for the output marginal pi_y
                                   # (0 -> uniform)

    # --- misc ----------------------------------------------------------------
    seed=1337,
    out_dir="data/synth_shuffled_x1_y0_2p17train_2p14val_V1024",
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
# Exact-count shuffled stream                                                  #
# --------------------------------------------------------------------------- #
def exact_counts(pi, n_tokens):
    """Largest-remainder apportionment of n_tokens to classes: counts sum to
    n_tokens and count_i is n_tokens*pi_i rounded, so the empirical marginal
    matches pi up to the unavoidable 1/n_tokens granularity."""
    ideal = pi * n_tokens
    counts = np.floor(ideal).astype(np.int64)
    short = n_tokens - counts.sum()
    if short > 0:
        remainder = ideal - counts
        top = np.argsort(remainder)[::-1][:short]
        counts[top] += 1
    return counts


def shuffled_stream(pi, n_tokens, rng):
    """Multiset with exact per-class counts, in uniformly random order."""
    counts = exact_counts(pi, n_tokens)
    stream = np.repeat(np.arange(pi.shape[0], dtype=np.uint16), counts)
    return rng.permutation(stream)


def write_stream(prefix, n_tokens, pi_x, pi_y, rng, out_dir):
    """Write <prefix>_x.bin / <prefix>_y.bin: two independently shuffled
    exact-count multisets, paired by position."""
    x = shuffled_stream(pi_x, n_tokens, rng)
    y = shuffled_stream(pi_y, n_tokens, rng)

    x.tofile(os.path.join(out_dir, f"{prefix}_x.bin"))
    y.tofile(os.path.join(out_dir, f"{prefix}_y.bin"))
    print(f"[write] {prefix}_x.bin / {prefix}_y.bin  ({n_tokens:,} tokens each)")

    # self-check: empirical marginals should match the targets exactly
    # (up to the 1/n_tokens rounding of exact_counts)
    V = pi_x.shape[0]
    emp_x = np.bincount(x, minlength=V) / n_tokens
    emp_y = np.bincount(y, minlength=V) / n_tokens
    tv_x = 0.5 * np.abs(emp_x - pi_x).sum()
    tv_y = 0.5 * np.abs(emp_y - pi_y).sum()
    print(f"[check] {prefix}: TV(emp_x, pi_x) = {tv_x:.2e}  "
          f"TV(emp_y, pi_y) = {tv_y:.2e}  (~V/(2N) rounding floor)")
    return n_tokens


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def build(cfg):
    rng = np.random.default_rng(cfg["seed"])
    V = cfg["vocab_size"]

    for key in ("n_train_tokens", "n_val_tokens"):
        n = cfg[key]
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0 or n & (n - 1):
            raise ValueError(f"{key} must be a positive power of two, got {n!r}")

    pi_x = T.make_pi(V, kind="zipf", zipf_s=cfg["alpha_x"])
    pi_y = T.make_pi(V, kind="zipf", zipf_s=cfg["alpha_y"])
    print(f"[build] vocab={V}  alpha_x={cfg['alpha_x']}  alpha_y={cfg['alpha_y']}")
    print(f"[build] H(pi_x) = {-(pi_x * np.log(pi_x)).sum():.3f} nats  "
          f"H(pi_y) = {-(pi_y * np.log(pi_y)).sum():.3f} nats  "
          f"(max = {np.log(V):.3f})")

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    n_tr = write_stream("train", cfg["n_train_tokens"], pi_x, pi_y, rng, out_dir)
    n_va = write_stream("val", cfg["n_val_tokens"], pi_x, pi_y, rng, out_dir)

    meta = dict(
        vocab_size=V,
        pi=pi_x,               # input marginal (pi_x)
        pi_y=pi_y,             # output marginal
        P=None,                # no Markov chain in this construction
        K=None,                # no x->y conditional kernel: pairing is random
        label_mode="shuffled_marginals",
        dual_stream=True,
        n_train_tokens=n_tr,
        n_val_tokens=n_va,
        config=cfg,
        seed=cfg["seed"],
    )
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"[write] meta.pkl  -> {out_dir}")
    print("[done]")


if __name__ == "__main__":
    cfg = load_config(sys.argv[1:])
    build(cfg)
