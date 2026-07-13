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

    # --- knob 1: token-frequency distribution pi (balance vs imbalance) ------
    freq="zipf",                   # "uniform" | "zipf" | "real"
    zipf_s=1.0,                    # zipf exponent (only if freq == "zipf")
    real_counts_path=None,         # .npy of length vocab_size (only if "real")

    # --- knob 2: predictability / difficulty (DECOUPLED from pi) -------------
    predictability=0.8,            # in [0, 1]; 0 = unigram-hard, 1 = sharp-easy
    bandwidth_frac=0.02,           # structural sharpness of learnable component

    # --- knob 3: label construction (default keeps y = shifted input) --------
    label_mode="shift",            # "shift" (default) | reserved for future
    shift=1,                       # next-token offset for label_mode == "shift"

    # --- misc ----------------------------------------------------------------
    seq_len=1024,                  # sequence length used only for sampling
                                   # continuity; the streams are flat afterwards
    order=1,                       # bigram; >1 reserved / not implemented yet
    seed=1337,
    out_dir="data/synth_zipf",
)


# --------------------------------------------------------------------------- #
# Target construction  (THE extension point for input-independent outputs)     #
# --------------------------------------------------------------------------- #
def build_targets(x, cfg):
    """Produce the target stream y from the input stream x.

    Default ("shift"): y[t] = x[t + shift]  -- i.e. standard next-token
    prediction, numerically identical to the original single-stream format.

    To later make outputs INDEPENDENT of the input (a different specified
    distribution / mapping), add a new branch here, e.g.:

        elif cfg["label_mode"] == "relabel":
            # deterministic token->token remap drawn from a chosen distribution
            y = label_map[x]
        elif cfg["label_mode"] == "sample":
            # sample y_t ~ p(y | x_t) from a separate output kernel
            y = sample_from_output_kernel(x, output_kernel, rng)

    Everything downstream (dual-stream .bin + meta) already supports this; only
    this function decides how y relates to x.
    """
    mode = cfg["label_mode"]
    if mode == "shift":
        s = cfg["shift"]
        # y is x shifted left by s; drop the last s tokens that have no target
        y = x[s:]
        x = x[:-s]
        return x, y
    raise NotImplementedError(
        f"label_mode='{mode}' not implemented yet. Default is 'shift'. "
        f"Add a branch in build_targets() to support input-independent outputs."
    )


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

    # 1. stationary token frequency pi  ------------------------------------- #
    real_counts = None
    if cfg["freq"] == "real":
        real_counts = np.load(cfg["real_counts_path"])
    pi = T.make_pi(V, kind=cfg["freq"], zipf_s=cfg["zipf_s"],
                   real_counts=real_counts)

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

    # 3. sample the flat token streams  ------------------------------------ #
    def make_stream(n_tokens, rng):
        # sample in seq_len chunks (each chunk stationary) then concatenate
        L = cfg["seq_len"]
        n_chunks = int(np.ceil(n_tokens / L))
        chunks = [T.sample_sequence(P, pi, L, rng) for _ in range(n_chunks)]
        return np.concatenate(chunks)[:n_tokens]

    print("[build] sampling train stream ...")
    x_train = make_stream(cfg["n_train_tokens"], rng)
    print("[build] sampling val stream ...")
    x_val = make_stream(cfg["n_val_tokens"], rng)

    # 4. build (x, y) dual streams per label_mode  ------------------------- #
    xtr, ytr = build_targets(x_train, cfg)
    xva, yva = build_targets(x_val, cfg)

    # 5. write to disk  ---------------------------------------------------- #
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in [("train_x", xtr), ("train_y", ytr),
                      ("val_x", xva), ("val_y", yva)]:
        arr.astype(np.uint16).tofile(os.path.join(out_dir, f"{name}.bin"))
        print(f"[write] {name}.bin  ({arr.size:,} tokens)")

    meta = dict(
        vocab_size=V,
        pi=pi,
        P=P,
        stationary_tv=tv,
        row_entropy_mean=float(ent.mean()),
        label_mode=cfg["label_mode"],
        dual_stream=True,
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
