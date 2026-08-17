#!/usr/bin/env python
"""Recompute the 2D preconditioned-Hessian heterogeneity trajectory with the
WITHIN-block axis measured as std(log10 eigenvalue) instead of spectral entropy,
reading the eigenvalue arrays already saved by compute_precond_hessian.py (no
SCO rerun needed).

Both axes are then std(log10 ...): X = std over blocks of log10(per-block mean
eigenvalue); Y = mean over blocks of [std over that block's log10 eigenvalues].
Larger = more heterogeneous on both axes (opposite direction from spectral
entropy, where larger = more uniform).

Effective-spectrum floor: eigenvalues below lambda_max * REL_FLOOR (default
1e-6) are numerical zeros (see the ~1e-4 -> 1e-14 cliff in the spectra) and are
dropped before taking log10, else they blow up the std.

Reads:  <run>/precond_hessian/<layer>_<optim>/{<tag>_eigs.npy, <tag>_block_mean.npy}
Writes: files/precond_2d_<group>_stdlog/precond_2d_<layer>.png (+ csv)

Usage:
    python plot_precond_2d_stdlog.py --group frz_embd
    python plot_precond_2d_stdlog.py --group frz_lmhead
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
TAG_ORDER = ["init", "p10", "p25", "p40", "p50", "p60", "p75", "p85", "p100"]
LAYERS = ["lm_head", "embedding"]
REL_FLOOR = 1e-6                       # drop eigs below lambda_max * REL_FLOOR

OPTIM_CMAP = {
    "sgd":  LinearSegmentedColormap.from_list("sgd_blue",  ["#bcd4f0", "#08306b"]),
    "adam": LinearSegmentedColormap.from_list("adam_red",  ["#f4b6ac", "#67000d"]),
    "muon": LinearSegmentedColormap.from_list("muon_green", ["#b7e2b1", "#00441b"]),
}
OPTIM_DARK = {"sgd": "#08306b", "adam": "#67000d", "muon": "#00441b"}
STYLE = {
    "sgd":  dict(marker="o", s=170, z=3, ring="#08306b"),
    "muon": dict(marker="s", s=95,  z=4, ring="#00441b"),
    "adam": dict(marker="D", s=55,  z=5, ring="#67000d"),
}
ORDER = {"sgd": 0, "muon": 1, "adam": 2}

# same run groups as submit_sco_precond.py
GROUPS = {
    "frz_embd": [
        ("REP-mserep-pos0-frz_embd-fullbs-sgd-lr0p048-imb-initG02-nobias", "sgd"),
        ("REP-mserep-pos0-frz_embd-fullbs-adam-lr3e-6-imb-initG02-nobias", "adam"),
        ("REP-muon-lr6e-5-G02-mom0", "muon"),
    ],
    "frz_lmhead": [
        ("REP1-frz_lmhead-sgd-lr0p01", "sgd"),
        ("REP1-frz_lmhead-adam-lr2e-5-G02", "adam"),
        ("REP1-frz_lmhead-muon-lr1e-4-G02-mom0", "muon"),
    ],
}


def std_log10_within(eig_rows):
    """eig_rows: (k, d) or (d,) eigenvalues of one or more blocks. Return the
    MEAN over blocks of std(log10 lambda) using only the effective spectrum
    (lambda > lambda_max * REL_FLOOR per block)."""
    rows = np.atleast_2d(eig_rows)
    vals = []
    for row in rows:
        row = np.asarray(row, float)
        pos = row[row > 0]
        if pos.size < 2:
            continue
        keep = pos[pos > pos.max() * REL_FLOOR]
        if keep.size < 2:
            continue
        vals.append(np.std(np.log10(keep)))
    return float(np.mean(vals)) if vals else np.nan


def load_run_layer(run_dir, layer):
    base = os.path.join(run_dir, "precond_hessian")
    if not os.path.isdir(base):
        return None
    sub = next((os.path.join(base, n) for n in os.listdir(base)
                if n.startswith(layer + "_")), None)
    if sub is None:
        return None
    # need per-tag eigs (within) + block_mean (between)
    summ = os.path.join(sub, "all_summary.json")
    optim = json.load(open(summ))[0]["optim"] if os.path.exists(summ) else "?"
    tags, xs, ys = [], [], []
    for tag in TAG_ORDER:
        ef = os.path.join(sub, f"{tag}_eigs.npy")
        bf = os.path.join(sub, f"{tag}_block_mean.npy")
        if not os.path.exists(ef):
            continue
        eigs = np.load(ef)
        y = std_log10_within(eigs)                       # within-block std(log10)
        # between-block: std(log10 per-block mean eigenvalue)
        if os.path.exists(bf):
            bm = np.load(bf)
            pos = bm[bm > 0]
            x = float(np.std(np.log10(pos))) if pos.size > 1 else 0.0
        else:
            x = 0.0
        if not np.isfinite(y):
            continue
        tags.append(tag); xs.append(x); ys.append(y)
    if not tags:
        return None
    return tags, np.array(xs), np.array(ys), optim


def plot_layer(runs, layer, out_path, csv_path):
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    results = []
    for run, _ in runs:
        run_dir = os.path.join(HERE, "runs", run)
        res = load_run_layer(run_dir, layer)
        if res is None:
            print(f"  [{layer}] no data for {run}")
            continue
        results.append(res)
    if not results:
        plt.close(fig); print(f"  [{layer}] nothing to plot"); return
    results.sort(key=lambda r: ORDER.get(r[3], 9))

    csv_rows = []
    for tags, xs, ys, optim in results:
        cmap = OPTIM_CMAP.get(optim, plt.get_cmap("Greys"))
        dark = OPTIM_DARK.get(optim, "#000")
        st = STYLE.get(optim, dict(marker="o", s=90, z=3, ring=dark))
        prog = np.linspace(0, 1, len(tags))
        if len(tags) >= 2:
            pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            ax.add_collection(LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 1),
                              array=prog[:-1], linewidth=2.0, zorder=st["z"] - 0.5))
        ax.scatter(xs, ys, c=prog, cmap=cmap, norm=plt.Normalize(0, 1),
                   s=st["s"], marker=st["marker"], edgecolor=st["ring"],
                   linewidth=1.3, zorder=st["z"])
        ax.plot([], [], marker=st["marker"], color=dark, label=optim.upper(),
                markersize=8, linestyle="-")
        ax.annotate("init", (xs[0], ys[0]), color=dark, fontsize=8, fontweight="bold",
                    textcoords="offset points", xytext=(5, 5))
        ax.annotate("p100", (xs[-1], ys[-1]), color=dark, fontsize=8, fontweight="bold",
                    textcoords="offset points", xytext=(5, -10))
        for t, x, y in zip(tags, xs, ys):
            csv_rows.append([layer, optim, t, x, y])

    ax.set_xlabel("BETWEEN-block heterogeneity\n"
                  r"std of $\log_{10}$(per-block mean eigenvalue of $P^{-1}H$)")
    ax.set_ylabel("WITHIN-block heterogeneity\n"
                  r"mean over blocks of std($\log_{10}\lambda$) of $P^{-1}H$ block")
    ax.set_title(f"Preconditioned-Hessian block heterogeneity  [{layer}]  (std-log10 both axes)\n"
                 r"SGD=blue, Adam=red, Muon=green; light$\to$dark = init$\to$p100")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3); ax.margins(0.12)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  wrote {out_path}")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "optim", "tag", "X_between_stdlog", "Y_within_stdlog"])
        w.writerows(csv_rows)
    print(f"  wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS) + ["all"], default="all")
    args = ap.parse_args()
    groups = list(GROUPS) if args.group == "all" else [args.group]
    for g in groups:
        out_dir = os.path.join(HERE, "files", f"precond_2d_{g}_stdlog")
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== {g} ===")
        for layer in LAYERS:
            plot_layer(GROUPS[g], layer,
                       os.path.join(out_dir, f"precond_2d_{layer}.png"),
                       os.path.join(out_dir, f"precond_2d_{layer}.csv"))


if __name__ == "__main__":
    main()
