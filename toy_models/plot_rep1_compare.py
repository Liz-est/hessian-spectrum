"""
plot_rep1_compare_corrected.py
================================
Compare the best REP1 runs (frozen lm_head, G02 init) across three optimizers:
- SGD lr=0.01 (val best)
- Adam lr=2e-5 (val best)
- Muon mom=0 lr=1e-4 (train best)

Order: SGD, Adam, Muon (left to right).

Usage:
    python3 plot_rep1_compare_corrected.py
"""

import csv
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_rep_groups import (N_GROUPS, RUNS_DIR, freq_groups, load_pair_stats)

DATASET = "synth_zipf_imbalanced_s1_V10000"
FLOOR_CLIP = 1e-10

# Order: SGD, Adam, Muon
RUNS = [
    "REP1-frz_lmhead-sgd-lr0p01",
    "REP1-frz_lmhead-adam-lr2e-5-G02",
    "REP1-frz_lmhead-muon-lr1e-4-G02-mom0",
]


def opt_label(exp):
    """Human-readable optimizer label from experiment config."""
    o, lr = exp["optim"], exp["lr"]
    name = o["name"]
    if name in ("adam", "adamw"):
        label = f"Adam betas={tuple(o['betas'])}"
    elif name == "muon":
        label = f"Muon momentum={o.get('muon_momentum', 0.95)}"
    else:
        label = f"SGD momentum={o.get('momentum', 0)}"
    label += f"  lr={lr['learning_rate']:g}  wd={o['weight_decay']:g}"
    return label


def load_run(name):
    csv_path = os.path.join(RUNS_DIR, name, "rep_groups.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"missing {csv_path}")
    with open(csv_path) as f:
        rows = sorted(list(csv.reader(f))[1:], key=lambda r: int(r[1]))
    its = np.array([int(r[1]) for r in rows])
    total = np.array([float(r[2]) for r in rows])
    gls = np.array([[float(v) for v in r[3:]] for r in rows])

    # read optimizer config from init checkpoint
    ck = torch.load(os.path.join(RUNS_DIR, name, "ckpt_init.pt"),
                    map_location="cpu", weights_only=False)
    exp = ck["experiment"]
    del ck
    label = opt_label(exp)

    return {"name": name, "its": its, "total": total, "gls": gls,
            "final": float(total[-1]), "label": label}


def main():
    ps = load_pair_stats(DATASET)
    groups = freq_groups(ps["pi"], N_GROUPS)
    rho = ps["rho"]
    # loss_type "mse_rep": loss = 0.5 * sum over classes, scale = 1.0
    gfloor = np.array([ps["floor"][g].sum() / max(rho[g].sum(), 1e-300)
                       for g in groups])
    floor_total = float(ps["floor"].sum())
    del ps

    print(f"floor_total = {floor_total:.6e}")
    data = []
    for name in RUNS:
        r = load_run(name)
        print(f"\n{r['label']}")
        print(f"  final={r['final']:.4e}  excess={r['final'] - floor_total:.4e}")
        gexcess = r["gls"][-1] - gfloor
        print("  final group excess G0..G9: "
              + " ".join(f"{v:.2e}" for v in gexcess))
        data.append(r)

    ng = len(groups)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharey=True)
    cmap = plt.cm.viridis

    for ax, r in zip(axes, data):
        for g in range(ng):
            ex = np.maximum(r["gls"][:, g] - gfloor[g], FLOOR_CLIP)
            ax.plot(r["its"], ex, marker="o", ms=3, lw=1.5,
                    color=cmap(g / max(1, ng - 1)),
                    label=f"G{g} ranks {groups[g][0]}-{groups[g][-1]}")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_title(f"{r['label']}\nfinal total {r['final']:.4e}", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)

    axes[0].set_ylabel("group excess loss (mse_rep, 0.5·sum)")
    axes[0].legend(fontsize=7, ncol=2, title="pi-mass groups (0 = frequent)",
                   title_fontsize=8)

    fig.suptitle(
        "REP1 frozen lm_head (G02 init): SGD vs Adam vs Muon\n"
        "per-frequency-group excess loss (loss − irreducible floor, "
        f"floor total = {floor_total:.4e})",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(RUNS_DIR, "rep1_compare_sgd_adam_muon_corrected.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {out}")

    # total excess loss overlay
    COLORS = {"sgd": "#ff7f0e", "adam": "#1f77b4", "muon": "#2ca02c"}
    plt.figure(figsize=(8, 5))
    for r in data:
        ex_total = np.maximum(r["total"] - floor_total, FLOOR_CLIP)
        opt = r["name"].split("-")[2]  # "sgd", "adam", or "muon"
        plt.plot(r["its"], ex_total, marker="o", ms=4, lw=1.8,
                 color=COLORS.get(opt, "#333"),
                 label=r["label"])
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("total excess loss (mse_rep, 0.5·sum)")
    plt.title("REP1 frozen lm_head (G02 init): total excess loss\n"
              f"(floor = {floor_total:.4e})", fontsize=11)
    plt.grid(alpha=0.25, lw=0.5)
    plt.legend(fontsize=7.5)
    plt.tight_layout()
    out2 = os.path.join(RUNS_DIR, "rep1_compare_total_corrected.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"[plot] {out2}")


if __name__ == "__main__":
    main()
