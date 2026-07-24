"""
build_dataset.py
================
Generate a synthetic bigram-language dataset for Hessian analysis and write it
in a *dual-stream* on-disk format that the (patched) trainer and Hessian
estimator can read:

    <out_dir>/
        train_x.bin   uint16  input  token ids
        train_y.bin   uint16  target token ids   (default: train_x shifted by 1)
        val_x.bin     uint16
        val_y.bin     uint16
        meta.pkl      dict with vocab_size, pi, P, config, seed, label_mode, ...

Why dual-stream?
----------------
The original NanoGPT format stores a single token stream and *derives* the
target as "input shifted by one" inside get_batch.  That hard-codes
"output == shifted input".  By materialising x and y as separate arrays we keep
that default (y is literally x shifted) yet leave a clean seam to later specify
an output distribution that is NOT a shift of the input -- just change
`label_mode` / `build_targets` below.  No downstream code needs to change.

Usage
-----
    python build_dataset.py configs/zipf_imbalanced.py
    python build_dataset.py configs/uniform_balanced.py out_dir=data/my_balanced

Any `key=value` on the command line overrides the config (parsed like the
NanoGPT `configurator.py` convention, but self-contained here).
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
    vocab_size=2048,
    n_train_tokens=10_000_000,     # total tokens in the training stream
    n_val_tokens=100_000,          # total tokens in the validation stream

    # --- knob 1: INPUT token-frequency distribution pi_x (balance vs imbalance)
    freq="zipf",                   # "uniform" | "zipf" | "real"
    zipf_s=1.0,                    # zipf exponent (only if freq == "zipf")
    real_counts_path=None,         # .npy of length vocab_size (only if "real")

    # --- knob 2: predictability / difficulty (DECOUPLED from pi) -------------
    predictability=0.8,            # in [0, 1]; 0 = unigram-hard, 1 = sharp-easy
    bandwidth_frac=0.02,           # structural sharpness of learnable component

    # --- knob 3: label construction (default keeps y = shifted input) --------
    label_mode="shift",            # "shift" | "independent" | "coupled"
    shift=1,                       # next-token offset for label_mode == "shift"

    # --- knob 3b: OUTPUT token-frequency distribution pi_y -------------------
    # Used when label_mode is "independent" or "coupled": y is emitted with the
    # SEPARATE output marginal pi_y.  These mirror the input knobs above.
    out_freq="uniform",            # "uniform" | "zipf" | "real"
    out_zipf_s=1.0,                # zipf exponent (only if out_freq == "zipf")
    out_real_counts_path=None,     # .npy of length vocab_size (only if "real")

    # --- knob 3c: x-y DEPENDENCE (only for label_mode == "coupled") ----------
    # "coupled" gives x and y DIFFERENT marginals (pi_x, pi_y) yet makes them
    # DEPENDENT: y_t ~ K(x_t, :) with pi_x^T K == pi_y held exact for any knob.
    coupling_strength=1.0,         # b in [0,1]: 0 -> independent, 1 -> full
                                   # structured dependence (mutual info grows w/ b)
    coupling_bandwidth_frac=0.02,  # sharpness of the x->y structure (near-diagonal
                                   # bump: x maps to y of a nearby index)

    # --- misc ----------------------------------------------------------------
    seq_len=1024,                  # sequence length used only for sampling
                                   # continuity; the streams are flat afterwards
    order=1,                       # bigram; >1 reserved / not implemented yet
    batch_chunks=4096,             # chunks sampled per batch; peak memory is
                                   # ~24 bytes * batch_chunks * seq_len
    seed=1337,
    out_dir="data/synth_zipf",
)


# --------------------------------------------------------------------------- #
# Target construction  (THE extension point for input-independent outputs)     #
# --------------------------------------------------------------------------- #
# Default ("shift"): y[t] = x[t + shift] -- standard next-token prediction,
# numerically identical to the original single-stream format.  The shift is
# applied *while streaming* in write_stream below: the y file starts `shift`
# tokens into the stream, and the x file holds back the last `shift` tokens,
# so both files end up n_tokens - shift long without ever materialising the
# full stream in memory.
#
# "independent" (IMPLEMENTED below): y_t is drawn i.i.d. from a SEPARATE output
# distribution pi_y (its own freq / zipf_s), with NO dependence on x whatsoever.
# x still carries its own input marginal pi_x via the bigram chain, so the input
# and output frequency distributions are fully decoupled -- e.g. x balanced /
# y Zipf, or x Zipf / y balanced.  Because y no longer comes from shifting x,
# there is no cross-batch carry: x and y streams have identical length and each
# batch is written independently (see write_stream_independent below).
#
# "coupled" (IMPLEMENTED below): x and y have DIFFERENT marginals (pi_x, pi_y)
# yet are DEPENDENT.  y_t ~ K(x_t, :) for a conditional kernel K built so the
# achieved output marginal pi_x^T K == pi_y is held EXACT for any dependence
# strength b (see transition.build_coupling_kernel).  b = 0 recovers the
# independent case; b = 1 is a structured (near-diagonal) dependence.  This is
# the "different marginals + not independent" cell.  Pointwise like the others
# -> no cross-batch carry; everything downstream already supports arbitrary y.


# --------------------------------------------------------------------------- #
# Config loading (config file first, then key=value overrides)                 #
# --------------------------------------------------------------------------- #
def load_config(argv):
    cfg = dict(CONFIG)
    for arg in argv:
        if arg.endswith(".py"):
            # execute the config file in a namespace and pull known keys
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
                v = ast.literal_eval(v)      # parse ints/floats/None/lists
            except (ValueError, SyntaxError):
                pass                          # keep as string
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

    if cfg["order"] != 1:
        raise NotImplementedError(
            "order > 1 (n-gram) is reserved but not implemented; use order=1."
        )

    # 1. INPUT stationary token frequency pi (== pi_x)  --------------------- #
    real_counts = None
    if cfg["freq"] == "real":
        real_counts = np.load(cfg["real_counts_path"])
    pi = T.make_pi(V, kind=cfg["freq"], zipf_s=cfg["zipf_s"],
                   real_counts=real_counts)

    # 1b. OUTPUT token frequency pi_y (for "independent" / "coupled" modes) -- #
    pi_y = None
    K = None                                    # x->y conditional (coupled only)
    if cfg["label_mode"] in ("independent", "coupled"):
        out_real_counts = None
        if cfg["out_freq"] == "real":
            out_real_counts = np.load(cfg["out_real_counts_path"])
        pi_y = T.make_pi(V, kind=cfg["out_freq"], zipf_s=cfg["out_zipf_s"],
                         real_counts=out_real_counts)

    if cfg["label_mode"] == "independent":
        print(f"[build] label_mode=independent  out_freq={cfg['out_freq']}"
              f"  out_zipf_s={cfg['out_zipf_s']}  "
              f"(y drawn i.i.d. from pi_y, independent of x)")
    elif cfg["label_mode"] == "coupled":
        b = cfg["coupling_strength"]
        K = T.build_coupling_kernel(pi, pi_y, strength=b,
                                    bandwidth_frac=cfg["coupling_bandwidth_frac"],
                                    rng=rng)
        pi_y_hat = T.output_marginal(pi, K)
        tv_y = 0.5 * np.abs(pi_y_hat - pi_y).sum()
        # Bayes-optimal loss for y | x = expected conditional row entropy under pi_x
        Hcond = float((pi * T.row_entropy(K)).sum())
        print(f"[build] label_mode=coupled  out_freq={cfg['out_freq']}"
              f"  out_zipf_s={cfg['out_zipf_s']}  strength(b)={b}")
        print(f"[build] achieved output-marginal TV(pi_y_hat, pi_y) = {tv_y:.2e}"
              f"  (should be ~0 for any b)")
        print(f"[build] H(y|x) = {Hcond:.3f} nats  (b=0 -> H(pi_y)={{:.3f}}; "
              f"lower = more x->y dependence)".format(
                  -(pi_y[pi_y > 0] * np.log(pi_y[pi_y > 0])).sum()))

    # 2. transition matrix P with stationary == pi, difficulty = predictability
    P = T.build_transition(pi, predictability=cfg["predictability"],
                           bandwidth_frac=cfg["bandwidth_frac"], rng=rng)

    # sanity check: does P actually have stationary distribution pi?
    pi_hat = T.stationary_distribution(P)
    tv = 0.5 * np.abs(pi_hat - pi).sum()          # total-variation distance
    ent = T.row_entropy(P)
    print(f"[build] vocab={V}  freq={cfg['freq']}  predictability={cfg['predictability']}")
    print(f"[build] stationary TV(pi_hat, pi) = {tv:.2e}  (should be ~0)")
    print(f"[build] row entropy: mean={ent.mean():.3f} nats  "
          f"(max possible={np.log(V):.3f})")

    # 3+4+5. sample, build (x, y), and write -- all streaming  ------------- #
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    n_tr = write_stream("train", cfg["n_train_tokens"], cfg, P, pi, pi_y, K,
                        rng, out_dir)
    n_va = write_stream("val", cfg["n_val_tokens"], cfg, P, pi, pi_y, K,
                        rng, out_dir)

    meta = dict(
        vocab_size=V,
        pi=pi,                 # input marginal (pi_x)
        pi_y=pi_y,             # output marginal (None unless independent/coupled)
        P=P,
        K=K,                   # x->y conditional kernel (None unless coupled)
        stationary_tv=tv,
        row_entropy_mean=float(ent.mean()),
        label_mode=cfg["label_mode"],
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


def write_stream(prefix, n_tokens, cfg, P, pi, pi_y, K, rng, out_dir):
    """Dispatch to the right label-mode writer for <prefix>_x.bin/_y.bin.

    All modes sample the input stream x in `batch_chunks`-sized chunks with
    sample_sequences_batch and append straight to disk as uint16, so peak
    memory is O(batch_chunks * seq_len) regardless of n_tokens.  They differ
    only in how the target stream y is produced.
    """
    mode = cfg["label_mode"]
    if mode == "shift":
        return write_stream_shift(prefix, n_tokens, cfg, P, pi, rng, out_dir)
    elif mode == "independent":
        return write_stream_independent(prefix, n_tokens, cfg, P, pi, pi_y,
                                        rng, out_dir)
    elif mode == "coupled":
        return write_stream_coupled(prefix, n_tokens, cfg, P, pi, K,
                                    rng, out_dir)
    else:
        raise NotImplementedError(
            f"label_mode='{mode}' not implemented. Supported: 'shift' "
            f"(y = input shifted by `shift`), 'independent' (y i.i.d. from a "
            f"separate pi_y, independent of x), and 'coupled' (y ~ K(x,:) with "
            f"different marginals pi_x/pi_y but x-y dependent)."
        )


def write_stream_shift(prefix, n_tokens, cfg, P, pi, rng, out_dir):
    """label_mode 'shift': y = input shifted by `shift`.

    Applied at the stream level, identical to the old materialise-then-slice
    code: y = stream[shift:], x = stream[:-shift].  Across batches that means
    the y file skips the first `shift` tokens of the stream and the x file
    holds back a `shift`-token carry until the end.
    """
    s = cfg["shift"]
    L = cfg["seq_len"]
    n_chunks = int(np.ceil(n_tokens / L))
    batch = max(1, int(cfg["batch_chunks"]))

    path_x = os.path.join(out_dir, f"{prefix}_x.bin")
    path_y = os.path.join(out_dir, f"{prefix}_y.bin")
    remaining = n_tokens
    seen = 0                                   # tokens emitted so far
    carry = np.empty(0, dtype=np.uint16)       # tail held back from x
    with open(path_x, "wb") as fx, open(path_y, "wb") as fy:
        for start in range(0, n_chunks, batch):
            b = min(batch, n_chunks - start)
            seg = T.sample_sequences_batch(P, pi, b, L, rng).ravel()
            seg = seg[:remaining].astype(np.uint16)
            remaining -= seg.size

            # y = stream[s:]  -> skip the first s tokens of the whole stream
            skip = max(0, s - seen)
            seg[skip:].tofile(fy)

            # x = stream[:-s] -> always hold back the latest s tokens
            buf = np.concatenate([carry, seg])
            buf[:max(0, buf.size - s)].tofile(fx)
            carry = buf[buf.size - min(s, buf.size):]

            seen += seg.size
            print(f"[write] {prefix}: {seen:,} / {n_tokens:,} tokens sampled",
                  flush=True)

    n_written = max(0, n_tokens - s)
    print(f"[write] {prefix}_x.bin / {prefix}_y.bin  ({n_written:,} tokens each)")
    return n_written


def write_stream_independent(prefix, n_tokens, cfg, P, pi, pi_y, rng, out_dir):
    """label_mode 'independent': y_t ~ i.i.d. pi_y, independent of x.

    x is still sampled from the bigram chain (so it keeps its own input
    marginal pi_x and the `predictability` structure), but each target token is
    drawn independently from the separate output distribution pi_y.  There is
    no shift and no cross-batch carry, so x and y streams have identical length
    (exactly n_tokens): every batch writes matched x/y chunks straight to disk.
    """
    L = cfg["seq_len"]
    n_chunks = int(np.ceil(n_tokens / L))
    batch = max(1, int(cfg["batch_chunks"]))
    V = pi.shape[0]

    path_x = os.path.join(out_dir, f"{prefix}_x.bin")
    path_y = os.path.join(out_dir, f"{prefix}_y.bin")
    remaining = n_tokens
    seen = 0
    with open(path_x, "wb") as fx, open(path_y, "wb") as fy:
        for start in range(0, n_chunks, batch):
            b = min(batch, n_chunks - start)
            seg_x = T.sample_sequences_batch(P, pi, b, L, rng).ravel()
            seg_x = seg_x[:remaining].astype(np.uint16)
            # y is independent of x: draw the SAME number of tokens i.i.d. pi_y
            seg_y = rng.choice(V, size=seg_x.size, p=pi_y).astype(np.uint16)
            remaining -= seg_x.size

            seg_x.tofile(fx)
            seg_y.tofile(fy)

            seen += seg_x.size
            print(f"[write] {prefix}: {seen:,} / {n_tokens:,} tokens sampled",
                  flush=True)

    print(f"[write] {prefix}_x.bin / {prefix}_y.bin  ({n_tokens:,} tokens each)")
    return n_tokens


def write_stream_coupled(prefix, n_tokens, cfg, P, pi, K, rng, out_dir):
    """label_mode 'coupled': y_t ~ K(x_t, :), different marginals but dependent.

    x is sampled from the bigram chain (input marginal pi_x); each target token
    is then drawn from the conditional row K[x_t] built by
    transition.build_coupling_kernel, so the achieved output marginal is exactly
    pi_y yet y depends on x (strength set by coupling_strength).  Like the
    independent mode there is no shift / no cross-batch carry: x and y streams
    have identical length (n_tokens) and each batch is written matched.
    """
    L = cfg["seq_len"]
    n_chunks = int(np.ceil(n_tokens / L))
    batch = max(1, int(cfg["batch_chunks"]))

    path_x = os.path.join(out_dir, f"{prefix}_x.bin")
    path_y = os.path.join(out_dir, f"{prefix}_y.bin")
    remaining = n_tokens
    seen = 0
    with open(path_x, "wb") as fx, open(path_y, "wb") as fy:
        for start in range(0, n_chunks, batch):
            b = min(batch, n_chunks - start)
            seg_x = T.sample_sequences_batch(P, pi, b, L, rng).ravel()
            seg_x = seg_x[:remaining]
            # y depends on x: draw y_t ~ K(x_t, :)
            seg_y = T.sample_y_given_x(K, seg_x, rng).astype(np.uint16)
            seg_x = seg_x.astype(np.uint16)
            remaining -= seg_x.size

            seg_x.tofile(fx)
            seg_y.tofile(fy)

            seen += seg_x.size
            print(f"[write] {prefix}: {seen:,} / {n_tokens:,} tokens sampled",
                  flush=True)

    print(f"[write] {prefix}_x.bin / {prefix}_y.bin  ({n_tokens:,} tokens each)")
    return n_tokens


if __name__ == "__main__":
    cfg = load_config(sys.argv[1:])
    build(cfg)
