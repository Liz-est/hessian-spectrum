"""
plot_compare.py
===============
Overlay the loss curves of all finished replication runs (Adam AND SGD) in one
figure, legend labelled with the full optimizer setting of each run.  Also
writes a per-frequency-group comparison grid (one panel per group, all runs
overlaid).

Usage
-----
    python plot_compare.py            # all runs found under runs/
    python plot_compare.py sgd        # only runs whose name contains "sgd"

Outputs: runs/compare_loss.png, runs/compare_groups.png
"""

import csv
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPL_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(REPL_ROOT, "runs")


def opt_desc(cfg: dict) -> str:
    if cfg["optimizer"] == "adam":
        d = (f"Adam lr={cfg['lr']:g} betas={tuple(cfg['betas'])} "
             f"eps={cfg['eps']:g} wd={cfg['weight_decay']:g}")
    else:
        d = (f"SGD lr={cfg['lr']:g} mom={cfg['momentum']:g} "
             f"wd={cfg['weight_decay']:g}")
    mode = cfg.get("grad_mode", "population")
    if mode == "stochastic":
        d += f" | stoch bs={cfg['batch_size']}"
    elif mode == "dataset":
        d += f" | dataset={cfg.get('dataset', '?')}"
    else:
        d += " | population grad"
    return d


def load_run(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "loss_log.csv")) as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    data = np.array([[float(v) for v in r] for r in rows])
    n_groups = len([h for h in header if h.startswith("group")])
    return cfg, data[:, 0], data[:, 1], data[:, 3:3 + n_groups]


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    runs = []
    for name in sorted(os.listdir(RUNS_DIR)):
        run_dir = os.path.join(RUNS_DIR, name)
        if filt not in name or ("smoke" in name and "smoke" not in filt):
            continue
        if not os.path.exists(os.path.join(run_dir, "loss_log.csv")):
            continue
        runs.append((name, *load_run(run_dir)))
    if not runs:
        sys.exit(f"no finished runs under {RUNS_DIR} matching '{filt}'")

    # color by optimizer family, shade by lr within the family;
    # linestyle: population = solid, stochastic = dashed
    fams = {"adam": [r for r in runs if r[1]["optimizer"] == "adam"],
            "sgd": [r for r in runs if r[1]["optimizer"] == "sgd"]}
    cmaps = {"adam": plt.cm.Blues, "sgd": plt.cm.Oranges}
    colors, styles = {}, {}
    for fam, rs in fams.items():
        lrs = sorted({r[1]["lr"] for r in rs})
        for r in rs:
            i = lrs.index(r[1]["lr"])
            colors[r[0]] = cmaps[fam](0.35 + 0.6 * i / max(1, len(lrs) - 1))
            styles[r[0]] = ("--" if r[1].get("grad_mode") == "stochastic"
                            else "-")

    # ---- total loss, all runs overlaid ---------------------------------- #
    plt.figure(figsize=(8, 5))
    for name, cfg, its, losses, _ in runs:
        plt.plot(its, losses, label=opt_desc(cfg), color=colors[name],
                 linestyle=styles[name])
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("population f(W)")
    dim = runs[0][1]["dim"]
    plt.title(f"RotatedMatrixBigramProblem dim={dim}: Adam vs SGD\n"
              f"(eval = population pi-weighted 0.5*per-class-sum sq. error; "
              f"solid = population grad, dashed = stochastic)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(RUNS_DIR, "compare_loss.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] {out}")

    # ---- per-frequency-group grid --------------------------------------- #
    n_groups = runs[0][4].shape[1]
    ncol = 5
    nrow = int(np.ceil(n_groups / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             sharex=True, sharey=True)
    for g in range(n_groups):
        ax = axes.flat[g]
        for name, cfg, its, _, gls in runs:
            ax.plot(its, gls[:, g], label=opt_desc(cfg), color=colors[name],
                    linestyle=styles[name], linewidth=1)
        ax.set_yscale("log")
        ax.set_title(f"freq group {g} (0 = most frequent)", fontsize=9)
    for ax in axes.flat[n_groups:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    fig.suptitle(f"dim={dim}: pi-weighted group loss, Adam vs SGD", fontsize=12)
    fig.supxlabel("iteration")
    fig.supylabel("group loss")
    fig.tight_layout()
    out = os.path.join(RUNS_DIR, "compare_groups.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
