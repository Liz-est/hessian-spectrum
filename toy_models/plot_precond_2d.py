#!/usr/bin/env python
"""Plot the 2D preconditioned-Hessian block-heterogeneity trajectory.

Reads precond_hessian/<layer>_<optim>/all_summary.json (written by
compute_precond_hessian.py) from each run dir, and draws, per layer, one figure
overlaying every run's trajectory:

    X (between-block): std of log10(per-block mean eigenvalue of P^{-1}H)
    Y (within-block):  mean spectral entropy of the per-block P^{-1}H spectrum
                       (1 = perfectly uniform / flat spectrum)

Each run is one colored polyline (init -> p100), markers per checkpoint.
Output: <first_run_dir>/../precond_2d_<layer>.png  (and a combined csv).

Usage:
    python plot_precond_2d.py runs/RUN_A runs/RUN_B runs/RUN_C
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

TAG_ORDER = ["init", "p10", "p25", "p40", "p50", "p60", "p75", "p85", "p100"]
LAYERS = ["lm_head", "embedding"]

# per-optimizer color family: light (init) -> dark (p100)
OPTIM_CMAP = {
    "sgd":  LinearSegmentedColormap.from_list("sgd_blue",  ["#bcd4f0", "#08306b"]),
    "adam": LinearSegmentedColormap.from_list("adam_red",  ["#f4b6ac", "#67000d"]),
    "muon": LinearSegmentedColormap.from_list("muon_green", ["#b7e2b1", "#00441b"]),
}
OPTIM_DARK = {"sgd": "#08306b", "adam": "#67000d", "muon": "#00441b"}


def load_run_layer(run_dir, layer):
    """Return (tags, xs, ys, optim) for one run+layer, or None if missing."""
    # find the <layer>_<optim> subdir
    base = os.path.join(run_dir, "precond_hessian")
    if not os.path.isdir(base):
        return None
    sub = None
    for name in os.listdir(base):
        if name.startswith(layer + "_"):
            sub = os.path.join(base, name)
            break
    if sub is None:
        return None
    sfile = os.path.join(sub, "all_summary.json")
    if not os.path.exists(sfile):
        # fall back to per-tag summaries
        rows = []
        for t in TAG_ORDER:
            f = os.path.join(sub, f"{t}_summary.json")
            if os.path.exists(f):
                rows.append(json.load(open(f)))
    else:
        rows = json.load(open(sfile))
    if not rows:
        return None
    order = {t: i for i, t in enumerate(TAG_ORDER)}
    rows.sort(key=lambda r: order.get(r["tag"], 999))
    tags = [r["tag"] for r in rows]
    xs = np.array([r["X_between"] for r in rows], float)
    ys = np.array([r["Y_within_specH"] for r in rows], float)
    optim = rows[0].get("optim", "?")
    return tags, xs, ys, optim


def plot_layer(run_dirs, layer, out_path, csv_path):
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    any_data = False
    csv_rows = []

    # marker style + draw order per optim so coincident points stay visible:
    # SGD drawn first & largest (bottom), Muon medium, Adam smallest on top.
    STYLE = {
        "sgd":  dict(marker="o", s=170, z=3, ring="#08306b"),
        "muon": dict(marker="s", s=95,  z=4, ring="#00441b"),
        "adam": dict(marker="D", s=55,  z=5, ring="#67000d"),
    }
    order = {"sgd": 0, "muon": 1, "adam": 2}

    results = []
    for run_dir in run_dirs:
        res = load_run_layer(run_dir, layer)
        if res is None:
            print(f"  [{layer}] no data for {run_dir}")
            continue
        results.append((run_dir, res))
    # draw SGD -> Muon -> Adam so smaller markers land on top
    results.sort(key=lambda rr: order.get(rr[1][3], 9))

    for run_dir, res in results:
        tags, xs, ys, optim = res
        any_data = True
        cmap = OPTIM_CMAP.get(optim, plt.get_cmap("Greys"))
        dark = OPTIM_DARK.get(optim, "#000000")
        st = STYLE.get(optim, dict(marker="o", s=90, z=3, ring=dark))
        prog = np.linspace(0.0, 1.0, len(tags))

        # gradient polyline: light (init) -> dark (p100)
        if len(tags) >= 2:
            pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 1),
                                array=prog[:-1], linewidth=2.0, zorder=st["z"] - 0.5)
            ax.add_collection(lc)
        # markers: light->dark shading, per-optim shape/size, colored ring so a
        # marker hidden under another optim's still shows its outline
        ax.scatter(xs, ys, c=prog, cmap=cmap, norm=plt.Normalize(0, 1),
                   s=st["s"], marker=st["marker"], edgecolor=st["ring"],
                   linewidth=1.3, zorder=st["z"])
        # one legend entry per optimizer (use the dark end as the swatch)
        ax.plot([], [], marker=st["marker"], color=dark, label=optim.upper(),
                markersize=8, linestyle="-")
        # label only the endpoints, in the optimizer's dark color
        ax.annotate("init", (xs[0], ys[0]), color=dark, fontsize=8, fontweight="bold",
                    textcoords="offset points", xytext=(5, 5))
        ax.annotate("p100", (xs[-1], ys[-1]), color=dark, fontsize=8, fontweight="bold",
                    textcoords="offset points", xytext=(5, -10))
        for t, x, y in zip(tags, xs, ys):
            csv_rows.append([layer, optim, os.path.basename(run_dir), t, x, y])

    if not any_data:
        plt.close(fig)
        print(f"  [{layer}] nothing to plot")
        return

    ax.set_xlabel("BETWEEN-block heterogeneity\n"
                  r"std of $\log_{10}$(per-block mean eigenvalue of $P^{-1}H$)")
    ax.set_ylabel("WITHIN-block heterogeneity\n"
                  r"mean spectral entropy of $P^{-1}H$ block (1 = uniform)")
    ax.set_title(f"Preconditioned-Hessian block heterogeneity  [{layer}]\n"
                 r"color: SGD=blue, Adam=red, Muon=green; light$\to$dark = init$\to$p100")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.margins(0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "optim", "run", "tag", "X_between", "Y_within_specH"])
        w.writerows(csv_rows)
    print(f"  wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out_dir", default=None,
                    help="where to write figures (default: toy_models/files/precond_2d)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(here, "files", "precond_2d")
    os.makedirs(out_dir, exist_ok=True)

    for layer in LAYERS:
        plot_layer(args.run_dirs, layer,
                   os.path.join(out_dir, f"precond_2d_{layer}.png"),
                   os.path.join(out_dir, f"precond_2d_{layer}.csv"))


if __name__ == "__main__":
    main()
