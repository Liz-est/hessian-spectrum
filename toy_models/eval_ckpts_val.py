"""
Post-hoc validation loss from saved checkpoints.

Older training runs only logged train loss to loss_log.csv; the val loss was
printed to stdout and lost. This script reconstructs a (sparse) val curve by
loading every ckpt_*.pt in runs/<run>/ and evaluating it on the val split of
the dataset recorded in the checkpoint's experiment config.

Per run it writes:
    runs/<run>/val_from_ckpts.csv        (tag, iter, val_loss)
    runs/<run>/loss_curve_with_val.png   (train curve from loss_log.csv + val points)

Usage:
    python3 eval_ckpts_val.py                 # all runs under runs/ with val data
    python3 eval_ckpts_val.py layer5-balance-adamw simpliest_balance-sgd
    python3 eval_ckpts_val.py --eval_iters=50 --batch_size=64

Checkpoints from the two trainers differ in one way that matters here:
train_simpliest_model.py's model (simpliest_model.py) does NOT add the
sinusoidal pos_enc in forward, while train_vanilla_transformer.py's model
(vanilla_model.py) does. The state dicts are indistinguishable (pos_enc is a
non-persistent buffer), so the model class is chosen by the experiment name
recorded in the checkpoint ("simpliest*" -> simpliest_model, else
vanilla_model). Runs without <data>/val_x.bin (e.g. fineweb10B) are skipped.
"""

import csv
import os
import sys
import time
from dataclasses import fields

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simpliest_model
import vanilla_model

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RUNS_DIR = os.path.join(HERE, "runs")

EVAL_ITERS = 50
BATCH_SIZE = 64
SEED = 1337


def eval_ckpt(ckpt_path, val_x, val_y, device, run):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # simpliest_* runs were trained without pos_enc in forward; the rest with it.
    mod = simpliest_model if run.startswith("simpliest") else vanilla_model
    cfg_dict = dict(ck["config"])
    cfg_dict["device"] = device
    known = {f.name for f in fields(mod.ToyVanillaConfig)}
    cfg = mod.ToyVanillaConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    model = mod.ToyVanilla(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    block = cfg.block_size
    g = torch.Generator().manual_seed(SEED)  # same batches for every ckpt/run
    losses = torch.zeros(EVAL_ITERS)
    with torch.no_grad():
        for k in range(EVAL_ITERS):
            ix = torch.randint(len(val_x) - block, (BATCH_SIZE,), generator=g)
            x = torch.stack([torch.from_numpy(val_x[i:i + block].astype(np.int64)) for i in ix]).to(device)
            y = torch.stack([torch.from_numpy(val_y[i:i + block].astype(np.int64)) for i in ix]).to(device)
            _, loss = model(x, y)
            losses[k] = loss.item()
    return ck["iter_num"], ck["tag"], losses.mean().item()


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
    if not os.path.exists(os.path.join(data_dir, "val_x.bin")):
        print(f"[{run}] {dataset} has no val_x.bin, skipping")
        return
    val_x = np.memmap(os.path.join(data_dir, "val_x.bin"), dtype=np.uint16, mode="r")
    val_y = np.memmap(os.path.join(data_dir, "val_y.bin"), dtype=np.uint16, mode="r")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = []
    for name in ckpts:
        t0 = time.time()
        it, tag, vloss = eval_ckpt(os.path.join(run_dir, name), val_x, val_y, device, run)
        rows.append((tag, it, vloss))
        print(f"[{run}] {tag:>5s} iter {it:>6d}  val loss {vloss:.4f}  ({time.time()-t0:.1f}s)")
    rows.sort(key=lambda r: r[1])

    csv_path = os.path.join(run_dir, "val_from_ckpts.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "iter", "val_loss"])
        w.writerows(rows)
    print(f"[{run}] wrote {csv_path}")

    plt.figure(figsize=(7, 4.5))
    log_path = os.path.join(run_dir, "loss_log.csv")
    if os.path.exists(log_path):
        with open(log_path) as f:
            tr = [(int(r["iter"]), float(r["train_loss"])) for r in csv.DictReader(f)]
        plt.plot([r[0] for r in tr], [r[1] for r in tr], lw=1.2, label="train")
    plt.plot([r[1] for r in rows], [r[2] for r in rows],
             lw=1.2, marker="o", ms=4, color="C1", label="val (from ckpts)")
    plt.xlabel("iteration"); plt.ylabel("loss (cross-entropy)")
    plt.title(f"{run} loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fig_path = os.path.join(run_dir, "loss_curve_with_val.png")
    plt.savefig(fig_path, dpi=150); plt.close()
    print(f"[{run}] wrote {fig_path}")


def main():
    global EVAL_ITERS, BATCH_SIZE
    runs = []
    for a in sys.argv[1:]:
        if a.startswith("--eval_iters="):
            EVAL_ITERS = int(a.split("=", 1)[1])
        elif a.startswith("--batch_size="):
            BATCH_SIZE = int(a.split("=", 1)[1])
        else:
            runs.append(a)
    if not runs:
        runs = sorted(d for d in os.listdir(RUNS_DIR)
                      if os.path.isdir(os.path.join(RUNS_DIR, d)))
    for run in runs:
        process_run(run)


if __name__ == "__main__":
    main()
