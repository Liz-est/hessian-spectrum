"""
plot_best_gauss.py
==================
Pick the best SGD and best Adam lr among the rep-gauss-* runs (Gaussian frozen
embedding, bigram data) and draw the two-panel per-frequency-group EXCESS loss
comparison, in the same format as runs/compare_groups_best_bigram.png.

"Best" = lowest final total loss among the runs that never spike (a spike is an
iterate whose loss exceeds 1.5x the running minimum while increasing); if every
run in a family spikes, the lowest final loss wins and the spike count is
reported in the title.

Excess = group loss - group floor, where the floor is the irreducible
per-input-token loss (optimal row = empirical P(y|x)), read from floor.npy.
The group floor is the rho-weighted average over the group, matching how
train.py averages the per-class losses.

Outputs:
    runs/compare_groups_best_gauss_bigram.png
    runs/compare_loss_gauss_bigram.png
    plus a leaderboard table on stdout

Usage:
    python plot_best_gauss.py
"""

import csv
import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import freq_groups

REPL_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(REPL_ROOT, "runs")
PREFIX = "rep-gauss-bigram-"


def load_run(name):
    d = os.path.join(RUNS_DIR, name)
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(d, "loss_log.csv")) as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    data = np.array([[float(v) for v in r] for r in rows])
    ng = len([h for h in header if h.startswith("group")])
    pi = np.load(os.path.join(d, "pi.npy"))
    rho = np.load(os.path.join(d, "rho.npy"))
    floor = np.load(os.path.join(d, "floor.npy"))
    groups = freq_groups(pi, ng)
    # group floor: rho-weighted mean of floor_x / rho_x over the group, i.e.
    # sum(floor_g) / sum(rho_g) -- same weighting train.py uses for the losses
    gfloor = np.array([floor[g].sum() / max(rho[g].sum(), 1e-300)
                       for g in groups])
    return {"name": name, "cfg": cfg, "its": data[:, 0], "loss": data[:, 1],
            "gls": data[:, 3:3 + ng], "groups": groups, "gfloor": gfloor,
            "floor_total": float(floor.sum())}


def n_spikes(loss, thresh=1.5):
    return sum(1 for i in range(2, len(loss))
               if loss[i] > thresh * loss[:i].min() and loss[i] > loss[i - 1])


def pick_best(runs, opt, iters=500):
    """Best lr within one optimizer family, restricted to a single run length
    (mixing 500- and 2000-iter runs would just pick the longest)."""
    fam = [r for r in runs if r["cfg"]["optimizer"] == opt
           and r["cfg"]["max_iters"] == iters]
    if not fam:
        sys.exit(f"no {opt} runs with max_iters={iters} under "
                 f"{RUNS_DIR}/{PREFIX}*")
    for r in fam:
        r["nspike"] = n_spikes(r["loss"])
        r["final"] = float(r["loss"][-1])
    clean = [r for r in fam if r["nspike"] == 0 and np.isfinite(r["final"])]
    pool = clean or [r for r in fam if np.isfinite(r["final"])]
    return min(pool, key=lambda r: r["final"]), fam


def desc(r):
    c = r["cfg"]
    if c["optimizer"] == "adam":
        return f"Adam lr={c['lr']:g} betas={tuple(c['betas'])}"
    return f"SGD lr={c['lr']:g} mom={c['momentum']:g}"


def two_panel(best_sgd, best_adam, out):
    ng = best_sgd["gls"].shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cmap = plt.cm.viridis
    # float32 training noise floor: once a group's excess reaches ~1e-6 the
    # difference loss - floor is at the precision limit and can even go
    # slightly negative, so clip there instead of pretending to resolve 1e-12
    FLOOR_CLIP = 1e-7
    for ax, r in zip(axes, [best_sgd, best_adam]):
        for g in range(ng):
            ex = np.maximum(r["gls"][:, g] - r["gfloor"][g], FLOOR_CLIP)
            ax.plot(r["its"], ex, lw=1.5, color=cmap(g / max(1, ng - 1)),
                    label=f"G{g} ranks {r['groups'][g][0]}-{r['groups'][g][-1]}")
        ax.axhline(FLOOR_CLIP, color="gray", ls=":", lw=0.9)
        ax.set_ylim(FLOOR_CLIP / 2, None)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        t = f"{desc(r)}\nfinal total {r['final']:.4e}"
        if r["nspike"]:
            t += f"  ({r['nspike']} spikes)"
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("group excess loss (loss - irreducible floor)")
    axes[0].legend(fontsize=7, ncol=2, title="pi-mass groups (0 = frequent)",
                   title_fontsize=8)
    fig.suptitle(
        "Gaussian frozen embedding (std 0.2), bigram data: best SGD vs best "
        f"Adam\nper-frequency-group excess loss (floor total "
        f"{best_sgd['floor_total']:.6f}; 10 equal-pi-mass groups; "
        "dotted = float32 resolution limit)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def loss_compare(runs, floor_total, out):
    plt.figure(figsize=(8, 5))
    cmaps = {"adam": plt.cm.Blues, "sgd": plt.cm.Oranges}
    for opt in ("adam", "sgd"):
        fam = sorted((r for r in runs if r["cfg"]["optimizer"] == opt),
                     key=lambda r: r["cfg"]["lr"])
        for i, r in enumerate(fam):
            plt.plot(r["its"], np.maximum(r["loss"] - floor_total, 1e-12),
                     color=cmaps[opt](0.35 + 0.6 * i / max(1, len(fam) - 1)),
                     lw=1.3, label=desc(r))
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("total excess loss (f(W) - floor)")
    plt.title("Gaussian frozen embedding, bigram data: lr sweep\n"
              f"(Blues = Adam, Oranges = SGD; floor = {floor_total:.6f})",
              fontsize=10)
    plt.grid(alpha=0.25, lw=0.5)
    plt.legend(fontsize=6.5, ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] {out}")


def main():
    names = sorted(d for d in os.listdir(RUNS_DIR)
                   if d.startswith(PREFIX)
                   and os.path.exists(os.path.join(RUNS_DIR, d, "loss_log.csv")))
    if not names:
        sys.exit(f"no {PREFIX}* runs with loss_log.csv under {RUNS_DIR}")
    runs = [load_run(n) for n in names]
    floor_total = runs[0]["floor_total"]
    short = [r for r in runs if r["cfg"]["max_iters"] == 500]
    best_sgd, sgd_fam = pick_best(runs, "sgd")
    best_adam, adam_fam = pick_best(runs, "adam")

    print(f"floor_total = {floor_total:.6f}")
    for r in sorted(runs, key=lambda r: (r["cfg"]["max_iters"],
                                         r["cfg"]["optimizer"],
                                         r["cfg"]["lr"])):
        r.setdefault("nspike", n_spikes(r["loss"]))
        r.setdefault("final", float(r["loss"][-1]))
        ex = r["gls"][-1] - r["gfloor"]
        star = " *" if r in (best_sgd, best_adam) else ""
        print(f"{r['name']:<40s} final={r['final']:.6e} "
              f"excess={r['final'] - floor_total:.4e} "
              f"nspike={r['nspike']:<4d}{star}")
        print("     group excess G0..G9: "
              + " ".join(f"{v:.2e}" for v in ex))

    two_panel(best_sgd, best_adam,
              os.path.join(RUNS_DIR, "compare_groups_best_gauss_bigram.png"))
    loss_compare(short, floor_total,
                 os.path.join(RUNS_DIR, "compare_loss_gauss_bigram.png"))

    # 2000-iter runs, if present: same two-panel view at 4x the budget
    long_sgd = [r for r in runs if r["cfg"]["optimizer"] == "sgd"
                and r["cfg"]["max_iters"] == 2000]
    long_adam = [r for r in runs if r["cfg"]["optimizer"] == "adam"
                 and r["cfg"]["max_iters"] == 2000]
    if long_sgd and long_adam:
        bs, _ = pick_best(runs, "sgd", iters=2000)
        ba, _ = pick_best(runs, "adam", iters=2000)
        two_panel(bs, ba, os.path.join(
            RUNS_DIR, "compare_groups_best_gauss_bigram_it2000.png"))


if __name__ == "__main__":
    main()
