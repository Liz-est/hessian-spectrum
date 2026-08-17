"""
SGD vs AdamW train-loss comparison for the frozen-layer MSE experiments,
one figure per frozen layer, styled after l5_mse_train_loss.png
(left: full curve, log y; right: last 25% zoom, linear y).

    runs/mse0_frozen_embd-sgd-lr0p1-gradclip0-imbalance     vs
    runs/mse0_frozen_embd-adamw-lr1p5e-3-imbalance          -> frozen_embd_train_loss.png
    runs/mse0_frozen_lmhead-sgd-lr0p05-gradclip0-imbalance  vs
    runs/mse0_frozen_lmhead-adamw-lr1p5e-3-imbalance        -> frozen_lmhead_train_loss.png

Each output PNG is written into BOTH runs it compares.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

FIGS = [
    dict(
        title="mse0 frozen embedding (imbalance-s1): train loss",
        out="frozen_embd_train_loss.png",
        curves=[
            ("mse0_frozen_embd-sgd-lr0p1-gradclip0-imbalance",
             "SGD (no clip, lr 0.1)", "tab:cyan"),
            ("mse0_frozen_embd-adamw-lr1p5e-3-imbalance",
             "AdamW (lr 1.5e-3)", "tab:orange"),
        ],
    ),
    dict(
        title="mse0 frozen lm_head (imbalance-s1): train loss",
        out="frozen_lmhead_train_loss.png",
        curves=[
            ("mse0_frozen_lmhead-sgd-lr0p05-gradclip0-imbalance",
             "SGD (no clip, lr 0.05)", "tab:cyan"),
            ("mse0_frozen_lmhead-adamw-lr1p5e-3-imbalance",
             "AdamW (lr 1.5e-3)", "tab:orange"),
        ],
    ),
    dict(
        title="mse0-pos0 frozen embedding (imbalance-s1, no pos_enc): train loss",
        out="frozen_embd_pos0_train_loss.png",
        curves=[
            ("mse0-pos0-frozen_embd-sgd-lr0p1-imb-init1G",
             "SGD (lr 0.1)", "tab:cyan"),
            ("mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-init1G",
             "AdamW (lr 1.5e-3)", "tab:orange"),
        ],
    ),
    dict(
        title="mse0-pos0 frozen lm_head (imbalance-s1, no pos_enc): train loss",
        out="frozen_lmhead_pos0_train_loss.png",
        curves=[
            ("mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G",
             "SGD (lr 0.05)", "tab:cyan"),
            ("mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G",
             "AdamW (lr 1.5e-3)", "tab:orange"),
        ],
    ),
    dict(
        title="mse0-pos0 frozen embedding (imbalance-s1, no pos_enc, 1500 iters): train loss",
        out="frozen_embd_pos0_iter1500_train_loss.png",
        curves=[
            ("mse0-pos0-frozen_embd-sgd-lr0p1-imb-init1G-iter1500",
             "SGD (lr 0.1)", "tab:cyan"),
            ("mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-init1G-iter1500",
             "AdamW (lr 1.5e-3)", "tab:orange"),
        ],
    ),
    dict(
        title="mse0-pos0 frozen lm_head (imbalance-s1, no pos_enc, 1500 iters): train loss",
        out="frozen_lmhead_pos0_iter1500_train_loss.png",
        curves=[
            ("mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G-iter1500",
             "SGD (lr 0.05)", "tab:cyan"),
            ("mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G-iter1500",
             "AdamW (lr 1.5e-3)", "tab:orange"),
        ],
    ),
    dict(
        title="mse0-pos0 frozen embedding (imbalance-s1, no pos_enc, 1500 iters): train loss",
        out="frozen_embd_pos0_iter1500_lr_train_loss.png",
        curves=[
            ("mse0-pos0-frozen_embd-sgd-lr0p1-imb-init1G-iter1500",
             "SGD (lr 0.1)", "tab:cyan"),
            ("mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-init1G-iter1500",
             "AdamW (lr 1.5e-3)", "tab:orange"),
            ("mse0-pos0-frozen_embd-adamw-lr6e-3-imb-init1G-iter1500",
             "AdamW (lr 6e-3)", "tab:green"),
            ("mse0-pos0-frozen_embd-adamw-lr9e-3-imb-init1G-iter1500",
             "AdamW (lr 9e-3)", "tab:red"),
        ],
    ),
    dict(
        title="mse0-pos0 frozen lm_head (imbalance-s1, no pos_enc, 1500 iters): train loss",
        out="frozen_lmhead_pos0_iter1500_lr_train_loss.png",
        curves=[
            ("mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G-iter1500",
             "SGD (lr 0.05)", "tab:cyan"),
            ("mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G-iter1500",
             "AdamW (lr 1.5e-3)", "tab:orange"),
            ("mse0-pos0-frozen_lmhead-adamw-lr6e-3-imb-init1G-iter1500",
             "AdamW (lr 6e-3)", "tab:green"),
            ("mse0-pos0-frozen_lmhead-adamw-lr9e-3-imb-init1G-iter1500",
             "AdamW (lr 9e-3)", "tab:red"),
        ],
    ),
    dict(
        title="mse0-pos0 frozen lm_head (imbalance-s1, no pos_enc, full batch): train loss",
        out="frozen_lmhead_pos0_fullbs_train_loss.png",
        curves=[
            ("mse0-pos0-frz_lmhead-fullbs-sgd-imb",
             "SGD (lr 0.05)", "tab:cyan"),
            ("mse0-pos0-frz_lmhead-fullbs-adamw-imb",
             "AdamW (lr 6e-3)", "tab:orange"),
        ],
    ),
    dict(
        title="mse0-pos0 frozen lm_head (imbalance-s1, no pos_enc): train loss",
        out="frozen_lmhead_pos0_lr_train_loss.png",
        curves=[
            ("mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G",
             "SGD (lr 0.05)", "tab:cyan"),
            ("mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G",
             "AdamW (lr 1.5e-3)", "tab:orange"),
            ("mse0-pos0-frozen_lmhead-adamw-lr2e-3-imb-init1G",
             "AdamW (lr 2e-3)", "tab:green"),
            ("mse0-pos0-frozen_lmhead-adamw-lr1p5e-2-imb-init1G",
             "AdamW (lr 1.5e-2)", "tab:red"),
        ],
    ),
]


def read_loss(run):
    xs, ys = [], []
    with open(os.path.join(RUNS, run, "loss_log.csv")) as f:
        for row in csv.DictReader(f):
            xs.append(int(row["iter"]))
            ys.append(float(row["train_loss"]))
    return xs, ys


def main():
    for fig_spec in FIGS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        data = [(read_loss(run), label, color)
                for run, label, color in fig_spec["curves"]]

        for (xs, ys), label, color in data:
            ax1.plot(xs, ys, lw=1.5, color=color, label=label)
        ax1.set_yscale("log")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("MSE loss (log scale)")
        ax1.set_title(fig_spec["title"])
        ax1.grid(alpha=0.3, which="both", lw=0.5)
        ax1.legend()

        # zoom: last 25% of training, linear y
        for (xs, ys), label, color in data:
            cut = max(xs) * 3 // 4
            zx = [x for x in xs if x >= cut]
            zy = [y for x, y in zip(xs, ys) if x >= cut]
            ax2.plot(zx, zy, lw=1.2, marker="o", ms=2.5, color=color, label=label)
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("MSE loss")
        ax2.set_title("zoom: last 25% of training (linear y)")
        ax2.grid(alpha=0.3, lw=0.5)
        ax2.legend()
        plt.tight_layout()

        for run, _, _ in fig_spec["curves"]:
            out = os.path.join(RUNS, run, fig_spec["out"])
            fig.savefig(out, dpi=150)
            print("wrote", out)
        plt.close(fig)


if __name__ == "__main__":
    main()
