#!/usr/bin/env python
"""2D block-heterogeneity trajectory for a 2-layer (embed + lm_head) run.

Each checkpoint's Hessian for one layer is split into per-token blocks (the
block-diagonal indexed by parameter-matrix rows == tokens). We place the
checkpoint at a single 2D point:

  X (BETWEEN-block, "scale" heterogeneity):
      std of log10(per-block mean eigenvalue), over blocks with positive mean.
      Scale-invariant dispersion of block scales; uses ALL blocks (not just the
      two extremes) and varies smoothly over training.

  Y (WITHIN-block heterogeneity):
      for each block, normalize its eigenvalues to [-1,1] via (x-mean)/(max-min),
      histogram it and take JS-distance to N(0,1); average over blocks.
      Captures how non-uniform a single block's spectrum is.

Connecting the per-checkpoint points in training order gives one trajectory.

Reads eigs_<layer>.npy from each <tag>/ subdir of the run folder. Writes
block_hetero_2d_<layer>.png and block_hetero_2d.csv into the run folder.

NOTE on this run (mse_rep, 2-layer): lm_head curvature is class-independent, so
all its blocks are identical -> X and Y are trivially constant. Only `embedding`
carries signal, so it is the default layer. Pass --layer lm_head to force it.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

EPS = 1e-12
# checkpoint tags in training order (only those present are used)
TAG_ORDER = ["init", "p10", "p25", "p40", "p50", "p60", "p75", "p85", "p100"]


def gauss_pdf(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def js_distance(p, q):
    p = np.clip(p, EPS, None); q = np.clip(q, EPS, None)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(np.sqrt(max(js, 0.0)))


def between_block_x(block_means):
    """std of log10(positive block means). Scale-invariant block-scale spread."""
    pos = block_means[block_means > 0]
    if pos.size < 2:
        return 0.0
    return float(np.std(np.log10(pos)))


def within_block_y(eigs, n_bins=128, lo=-4.0, hi=4.0):
    """Mean over blocks of JS( normalized-block-spectrum , N(0,1) )."""
    edges = np.linspace(lo, hi, n_bins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    g = gauss_pdf(ctr, 0.0, 1.0)
    vals = []
    for row in eigs:
        row = np.clip(np.asarray(row, float), 0.0, None)
        rng = row.max() - row.min()
        if rng <= 0:
            continue  # degenerate block: no within-block structure
        z = (row - row.mean()) / rng
        h, _ = np.histogram(z, bins=edges, density=True)
        vals.append(js_distance(h + EPS, g + EPS))
    return float(np.mean(vals)) if vals else np.nan


def compute_point(eigs_path):
    eigs = np.load(eigs_path)                 # (n_blocks, k)
    bm = eigs.mean(axis=1)                     # per-block mean eigenvalue
    return between_block_x(bm), within_block_y(eigs), eigs.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--layer", default="embedding",
                    choices=["embedding", "lm_head"])
    args = ap.parse_args()

    tags, xs, ys = [], [], []
    for tag in TAG_ORDER:
        ef = os.path.join(args.run_dir, tag, f"eigs_{args.layer}.npy")
        if not os.path.exists(ef):
            continue
        x, y, n = compute_point(ef)
        if not np.isfinite(x) or not np.isfinite(y):
            # e.g. `init` is all-zeros -> undefined point, skip in the trajectory
            print(f"  skip {tag}: undefined (x={x}, y={y})")
            continue
        tags.append(tag); xs.append(x); ys.append(y)
        print(f"  {tag:5s}  X(between)={x:.4f}  Y(within)={y:.4f}  n_blocks={n}")

    if len(tags) < 1:
        raise SystemExit("no usable checkpoints found")

    xs = np.array(xs); ys = np.array(ys)

    # write csv
    csv_path = os.path.join(args.run_dir, f"block_hetero_2d_{args.layer}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "x_between_std_log10_blockmean", "y_within_js_gauss"])
        for t, x, y in zip(tags, xs, ys):
            w.writerow([t, x, y])

    # --- plot: trajectory colored by training progress ---
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    prog = np.linspace(0, 1, len(tags))
    if len(tags) >= 2:
        pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", array=prog[:-1],
                            linewidth=2.0, zorder=1)
        ax.add_collection(lc)
    sc = ax.scatter(xs, ys, c=prog, cmap="viridis", s=90, zorder=2,
                    edgecolor="black", linewidth=0.6)
    for t, x, y in zip(tags, xs, ys):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(6, 5),
                    fontsize=9)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("training progress (init → p100)")
    ax.set_xlabel("BETWEEN-block heterogeneity\nstd of log10(per-block mean eigenvalue)")
    ax.set_ylabel("WITHIN-block heterogeneity\nmean JS(normalized block spectrum, N(0,1))")
    ax.set_title(f"Block-Hessian heterogeneity trajectory  [{args.layer}]\n"
                 f"{os.path.basename(os.path.normpath(args.run_dir))}")

    # Auto-zoom each axis so a near-stationary trajectory (e.g. lm_head, whose
    # blocks are all identical -> x≡0 and y drifts only in the 4th decimal) is
    # still legible instead of collapsing to a single dot. Pad each axis by a
    # margin relative to its own span, with a floor so a truly-constant axis
    # still gets a visible window.
    def _lims(v):
        v = np.asarray(v, float)
        lo, hi = float(v.min()), float(v.max())
        span = hi - lo
        pad = max(span * 0.25, abs(hi) * 1e-3, 1e-4)
        return lo - pad, hi + pad
    ax.set_xlim(*_lims(xs))
    ax.set_ylim(*_lims(ys))
    ax.ticklabel_format(useOffset=False, style="plain", axis="both")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.run_dir, f"block_hetero_2d_{args.layer}.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
