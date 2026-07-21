"""
inspect_dataset.py
===================
Sanity-check a generated dataset:

  * empirical unigram frequency of the written stream  vs  the target pi
  * distribution of per-row transition entropy (the difficulty knob's effect)
  * a few basic dual-stream consistency checks

Usage
-----
    python inspect_dataset.py data/synth_zipf_imbalanced
    python inspect_dataset.py data/synth_zipf_imbalanced --no-plot

Produces `inspect.png` inside the dataset directory (unless --no-plot), and
prints a text summary that works headless.
"""

import os
import sys
import pickle

import numpy as np


def main(out_dir, do_plot=True):
    with open(os.path.join(out_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    V = meta["vocab_size"]
    pi = np.asarray(meta["pi"])
    P = np.asarray(meta["P"])

    # memmap instead of fromfile: the 1B-token streams are ~2GB each and
    # loading them wholesale gets the process OOM-killed
    x = np.memmap(os.path.join(out_dir, "train_x.bin"), dtype=np.uint16, mode="r")
    y_size = os.path.getsize(os.path.join(out_dir, "train_y.bin")) // np.dtype(np.uint16).itemsize

    # --- empirical unigram vs target pi (chunked bincount) ----------------- #
    counts = np.zeros(V, dtype=np.int64)
    chunk = 64 * 1024 * 1024  # 64M tokens = 128MB per read
    for i in range(0, x.size, chunk):
        counts += np.bincount(x[i:i + chunk], minlength=V)
    emp = counts.astype(np.float64)
    emp /= emp.sum()
    tv = 0.5 * np.abs(emp - pi).sum()

    # --- per-row entropy of P --------------------------------------------- #
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    ent = -(P * logP).sum(axis=1)

    # --- dual-stream consistency ------------------------------------------ #
    label_mode = meta.get("label_mode", "shift")
    consistent = "n/a"
    if label_mode == "shift":
        s = meta["config"].get("shift", 1)
        # for shift mode, y should equal x advanced by s (on the raw stream);
        # here we just report that the streams have the expected length offset
        consistent = f"shift={s}, len(x)={x.size:,}, len(y)={y_size:,}"

    print(f"=== inspect {out_dir} ===")
    print(f"vocab_size            : {V}")
    print(f"freq / predictability : {meta['config']['freq']} / "
          f"{meta['config']['predictability']}")
    print(f"target pi: max={pi.max():.4e} min={pi.min():.4e} "
          f"ratio={pi.max()/pi.min():.1f}")
    print(f"empirical vs pi  TV   : {tv:.4e}  (small => stream matches pi)")
    print(f"stationary check (meta): TV={meta.get('stationary_tv', float('nan')):.2e}")
    print(f"row entropy (nats)    : mean={ent.mean():.3f} "
          f"min={ent.min():.3f} max={ent.max():.3f}  (max possible={np.log(V):.3f})")
    print(f"label_mode            : {label_mode}  ({consistent})")

    if not do_plot:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available, skipping figure")
        return

    order = np.argsort(-pi)                       # sort by target frequency
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].loglog(np.arange(1, V + 1), pi[order], label="target pi")
    ax[0].loglog(np.arange(1, V + 1), emp[order], ".", ms=2, label="empirical")
    ax[0].set_title(f"unigram freq (TV={tv:.2e})")
    ax[0].set_xlabel("rank"); ax[0].set_ylabel("prob"); ax[0].legend()

    ax[1].hist(ent, bins=50)
    ax[1].axvline(np.log(V), color="r", ls="--", label="max entropy")
    ax[1].set_title("per-row transition entropy")
    ax[1].set_xlabel("nats"); ax[1].legend()

    # show a corner of P to eyeball its sharpness
    k = min(64, V)
    im = ax[2].imshow(P[:k, :k], aspect="auto", cmap="viridis")
    ax[2].set_title(f"P[:{k}, :{k}]")
    fig.colorbar(im, ax=ax[2])

    fig.tight_layout()
    path = os.path.join(out_dir, "inspect.png")
    fig.savefig(path, dpi=110)
    print(f"[plot] saved {path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_plot = "--no-plot" not in sys.argv
    if not args:
        print("usage: python inspect_dataset.py <out_dir> [--no-plot]")
        sys.exit(1)
    main(args[0], do_plot=do_plot)
