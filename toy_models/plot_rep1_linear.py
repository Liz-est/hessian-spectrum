"""
plot_rep1_linear.py -- linear-y version of the REP1 group comparison.
Top row: raw rho-weighted group loss.  Bottom row: excess (loss - floor).
Columns: SGD, Adam, Muon.  y-axis LINEAR (no log).
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_rep_groups import N_GROUPS, RUNS_DIR, freq_groups, load_pair_stats

DATASET = "synth_zipf_imbalanced_s1_V10000"
RUNS = [
    ("SGD lr=0.01",        "REP1-frz_lmhead-sgd-lr0p01"),
    ("Adam lr=2e-5",       "REP1-frz_lmhead-adam-lr2e-5-G02"),
    ("Muon lr=1e-4 mom0",  "REP1-frz_lmhead-muon-lr1e-4-G02-mom0"),
]


def load_run(name):
    with open(os.path.join(RUNS_DIR, name, "rep_groups.csv")) as f:
        rows = sorted(list(csv.reader(f))[1:], key=lambda r: int(r[1]))
    its = np.array([int(r[1]) for r in rows])
    gls = np.array([[float(v) for v in r[3:]] for r in rows])
    return its, gls


def main():
    ps = load_pair_stats(DATASET)
    groups = freq_groups(ps["pi"], N_GROUPS)
    rho = ps["rho"]
    gfloor = np.array([ps["floor"][g].sum() / max(rho[g].sum(), 1e-300)
                       for g in groups])
    ng = len(groups)
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for col, (title, name) in enumerate(RUNS):
        its, gls = load_run(name)
        for g in range(ng):
            c = cmap(g / max(1, ng - 1))
            lab = f"G{g} (ranks {groups[g][0]}-{groups[g][-1]})"
            axes[0, col].plot(its, gls[:, g], marker="o", ms=3, lw=1.5,
                              color=c, label=lab)
            axes[1, col].plot(its, gls[:, g] - gfloor[g], marker="o", ms=3,
                              lw=1.5, color=c, label=lab)
        axes[0, col].set_title(title, fontsize=11)
        for row in (0, 1):
            axes[row, col].grid(alpha=0.25, lw=0.5)
        axes[1, col].set_xlabel("iteration")

    axes[0, 0].set_ylabel("raw group loss (linear)")
    axes[1, 0].set_ylabel("excess = loss - floor (linear)")
    axes[0, 0].legend(fontsize=7, ncol=2, title="pi-mass groups (0=frequent)",
                      title_fontsize=8)
    fig.suptitle("REP1 frozen lm_head: per-frequency-group loss, LINEAR y-axis\n"
                 "top = raw loss (ranking set by floor), "
                 "bottom = excess above floor", fontsize=12)
    fig.tight_layout()
    out = os.path.join(RUNS_DIR, "rep1_linear_raw_vs_excess.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
