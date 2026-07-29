"""
Per-frequency-group validation loss from saved checkpoints.

Splits the vocab into 10 equal-size groups by token frequency (the dataset's
target unigram pi from meta.pkl, most-frequent group first), then for every
ckpt_*.pt in a run evaluates the FULL val split once and averages the
per-position cross-entropy by the frequency group of the target token. The
result is one val-loss curve per group, drawn with a sequential light->dark
blue ramp (dark = most frequent).

Per run it writes:
    runs/<run>/val_by_freq.csv        (tag, iter, group_0 .. group_9)
    runs/<run>/val_by_freq.png

Usage:
    python3 eval_ckpts_val_by_freq.py                        # all runs under runs/grid-sgd-1/
    python3 eval_ckpts_val_by_freq.py grid-sgd-1/l5-imb-sgd-lr0p1-wd0
    python3 eval_ckpts_val_by_freq.py --n_groups=10 --batch_size=64

Like eval_ckpts_val.py, the model class is chosen by the experiment name in
the checkpoint ("simpliest*" -> simpliest_model, else vanilla_model). Only
dual-stream datasets are supported (the group split needs the meta.pkl pi).
Supports both loss types: CE runs group the per-position cross-entropy; MSE
runs (loss_type="mse", e.g. the mse0_frozen_* freeze experiments) group the
per-position mean-squared error vs the one-hot target, so the group average
reproduces the training objective exactly.
"""

import csv
import os
import pickle
import sys
import time
from dataclasses import fields

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simpliest_model
import vanilla_model

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RUNS_DIR = os.path.join(HERE, "runs")

DEFAULT_GLOB_DIR = "grid-sgd-1"
N_GROUPS = 10
BATCH_SIZE = 64

# 10-step sequential blue ramp (OKLCH hue 259, monotone L, validated:
# adjacent dL >= 0.06, light end 2.03:1 on white). Index 0 = lightest.
RAMP = ["#86b5ff", "#619efe", "#4d89e7", "#3874d0", "#245fb9",
        "#0c4aa3", "#013885", "#002863", "#001943", "#000b27"]


def freq_groups(pi, n_groups):
    """Token ids split into n_groups equal-size groups, most frequent first.
    Returns (groups list, group_id per token)."""
    order = np.argsort(-pi)
    groups = np.array_split(order, n_groups)
    gid = np.empty(len(pi), dtype=np.int64)
    for i, g in enumerate(groups):
        gid[g] = i
    return groups, gid


def load_model(ck, run, device):
    mod = simpliest_model if run.startswith("simpliest") else vanilla_model
    cfg_dict = dict(ck["config"])
    cfg_dict["device"] = device
    known = {f.name for f in fields(mod.ToyVanillaConfig)}
    cfg = mod.ToyVanillaConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    model = mod.ToyVanilla(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def eval_ckpt_by_group(model, cfg, val_x, val_y, gid_t, device, n_groups):
    """One deterministic pass over the whole val split (non-overlapping
    blocks); returns the mean per-position loss per frequency group of the
    target token. The per-position loss matches the training objective:
    cross-entropy for loss_type="ce", mean-over-classes squared error vs the
    one-hot target for loss_type="mse" (so the all-group average equals the
    trainer's F.mse_loss)."""
    block = cfg.block_size
    loss_type = getattr(cfg, "loss_type", "ce")
    nseq = len(val_x) // block
    loss_sum = torch.zeros(n_groups, dtype=torch.float64)
    count = torch.zeros(n_groups, dtype=torch.float64)
    with torch.no_grad():
        for s in range(0, nseq, BATCH_SIZE):
            e = min(s + BATCH_SIZE, nseq)
            idx = np.arange(s, e) * block
            x = torch.stack([torch.from_numpy(val_x[i:i + block].astype(np.int64)) for i in idx]).to(device)
            y = torch.stack([torch.from_numpy(val_y[i:i + block].astype(np.int64)) for i in idx]).to(device)
            logits, _ = model(x, y)
            tgt = y.view(-1)
            flat = logits.view(-1, logits.size(-1))
            if loss_type == "mse":
                onehot = F.one_hot(tgt, num_classes=flat.size(-1)).to(flat.dtype)
                per_pos = (flat - onehot).pow(2).mean(dim=-1)  # mean over classes
            else:
                per_pos = F.cross_entropy(flat, tgt, reduction="none")
            g = gid_t[tgt]
            loss_sum.scatter_add_(0, g.cpu(), per_pos.double().cpu())
            count += torch.bincount(g.cpu(), minlength=n_groups).double()
    return (loss_sum / count).numpy(), count.numpy()


def plot_run(run, rows, masses, out_png, loss_type="ce"):
    iters = [r[1] for r in rows]
    plt.figure(figsize=(7.5, 5))
    ax = plt.gca()
    for g in range(N_GROUPS):
        # dark = most frequent (group 0)
        color = RAMP[N_GROUPS - 1 - g]
        vals = [r[2][g] for r in rows]
        label = f"G{g} ({100*masses[g]:.1f}%)"
        ax.plot(iters, vals, lw=1.8, marker="o", ms=3.5, color=color, label=label)
    # selective direct labels: most and least frequent group only
    ax.annotate(" G0 most freq", (iters[-1], rows[-1][2][0]), fontsize=8,
                color=RAMP[-1], va="center")
    ax.annotate(f" G{N_GROUPS-1} least freq", (iters[-1], rows[-1][2][-1]),
                fontsize=8, color=RAMP[2], va="center")
    ax.set_xlabel("iteration")
    if loss_type == "mse":
        ax.set_ylabel("val loss (MSE vs one-hot)")
        ax.set_yscale("log")   # zipf groups span decades in MSE; log keeps all visible
    else:
        ax.set_ylabel("val loss (cross-entropy)")
    ax.set_title(f"{run}\nval loss by token-frequency group (dark = frequent)")
    ax.grid(alpha=0.25, lw=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=7.5, ncol=2, title="group (pi mass)", title_fontsize=8,
              framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def process_run(run):
    run_dir = os.path.join(RUNS_DIR, run)
    ckpts = sorted(f for f in os.listdir(run_dir)
                   if f.startswith("ckpt_") and f.endswith(".pt"))
    if not ckpts:
        print(f"[{run}] no checkpoints, skipping")
        return

    ck0 = torch.load(os.path.join(run_dir, ckpts[0]), map_location="cpu", weights_only=False)
    dataset = (ck0.get("experiment") or {}).get("data", {}).get("dataset")
    if dataset is None:
        print(f"[{run}] no dataset recorded in checkpoint, skipping")
        return
    data_dir = os.path.join(REPO_ROOT, "data", dataset)
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(os.path.join(data_dir, "val_x.bin")):
        print(f"[{run}] {dataset} is not dual-stream, skipping")
        return
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    groups, gid = freq_groups(meta["pi"], N_GROUPS)
    masses = [float(meta["pi"][g].sum()) for g in groups]
    val_x = np.memmap(os.path.join(data_dir, "val_x.bin"), dtype=np.uint16, mode="r")
    val_y = np.memmap(os.path.join(data_dir, "val_y.bin"), dtype=np.uint16, mode="r")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gid_t = torch.from_numpy(gid).to(device)
    rows = []
    loss_type = "ce"
    for name in ckpts:
        t0 = time.time()
        ck = torch.load(os.path.join(run_dir, name), map_location="cpu", weights_only=False)
        model, cfg = load_model(ck, os.path.basename(run), device)
        loss_type = getattr(cfg, "loss_type", "ce")
        gl, cnt = eval_ckpt_by_group(model, cfg, val_x, val_y, gid_t, device, N_GROUPS)
        rows.append((ck["tag"], ck["iter_num"], gl))
        print(f"[{run}] {ck['tag']:>5s} iter {ck['iter_num']:>6d}  "
              f"G0 {gl[0]:.4g}  G{N_GROUPS-1} {gl[-1]:.4g}  ({loss_type}, {time.time()-t0:.1f}s)")
    rows.sort(key=lambda r: r[1])

    csv_path = os.path.join(run_dir, "val_by_freq.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "iter"] + [f"group_{g}" for g in range(N_GROUPS)])
        for tag, it, gl in rows:
            w.writerow([tag, it] + [f"{v:.6g}" for v in gl])
    print(f"[{run}] wrote {csv_path}")

    out_png = os.path.join(run_dir, "val_by_freq.png")
    plot_run(run, rows, masses, out_png, loss_type)
    print(f"[{run}] wrote {out_png}")


def main():
    global N_GROUPS, BATCH_SIZE
    runs = []
    for a in sys.argv[1:]:
        if a.startswith("--n_groups="):
            N_GROUPS = int(a.split("=", 1)[1])
        elif a.startswith("--batch_size="):
            BATCH_SIZE = int(a.split("=", 1)[1])
        else:
            runs.append(a)
    if not runs:
        base = os.path.join(RUNS_DIR, DEFAULT_GLOB_DIR)
        runs = sorted(os.path.join(DEFAULT_GLOB_DIR, d) for d in os.listdir(base)
                      if os.path.isdir(os.path.join(base, d)))
    for run in runs:
        process_run(run)


if __name__ == "__main__":
    main()
