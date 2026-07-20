"""
Train vanilla_transformer (single-layer vanilla decoder) on the synthetic bigram data,
log the loss curve, and checkpoint at 0% / 10% / 50% / 100% of training so the
Hessian analysis (analyze_vanilla.py) can be run at each stage.

Run from the toy_models/ directory:
    python3 train_vanilla_transformer.py                 # single process (CPU or 1 GPU)
    python3 train_vanilla_transformer.py --max_iters=40  # quick smoke test
    torchrun --standalone --nproc_per_node=8 train_vanilla_transformer.py   # 8-GPU DDP

The dataset is the V=1024 dual-stream synth data at
    <repo-root>/data/synth_uniform_balanced_V1024/
(token ids 0..1023, matching model C's vocab).

Under DDP the per-GPU batch is `batch_size`; the effective batch is
batch_size * world_size. Rank 0 evaluates, logs, and writes checkpoints.
"""

import os
import sys
import time
import math
import csv

# Cap CPU threads (only matters for the CPU path; harmless on GPU).
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vanilla_transformer import config_C
from vanilla_model import ToyVanilla

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# ----------------------------------------------------------------------------
# config (override any of these from the CLI as --key=value)
# ----------------------------------------------------------------------------
dataset = "synth_zipf_imbalanced_s1_V1024"
batch_size = 64
block_size = config_C.block_size          # 128, the model context length
max_iters = 8000                          # 100% of training
warmup_iters = 200
learning_rate = 6e-4
min_lr = 3e-5
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0
decay_lr = True
eval_interval = 200
eval_iters = 100
log_interval = 20
seed = 1337
out_dir = os.path.join(HERE, "runs", "vanilla_imbalance_s1")

# checkpoints at these fractions of training (init / 10% / 50% / 100%)
ckpt_fracs = {"init": 0.0, "p10": 0.10, "p50": 0.50, "p100": 1.0}

# ---- minimal CLI override (--key=value) ------------------------------------
for arg in sys.argv[1:]:
    assert arg.startswith("--") and "=" in arg, f"bad arg {arg}"
    key, val = arg[2:].split("=", 1)
    assert key in globals(), f"unknown config key {key}"
    cur = globals()[key]
    globals()[key] = type(cur)(val) if not isinstance(cur, bool) else (val == "True")


def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        if torch.cuda.is_available():
            local = int(os.environ.get("LOCAL_RANK", rank))
            torch.cuda.set_device(local)
            device = f"cuda:{local}"
        else:
            device = "cpu"
        return rank, world, device, True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return 0, 1, device, False


def main():
    rank, world, device, is_ddp = setup_dist()
    is_master = rank == 0
    # different seed per rank so each pulls different batches
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    if is_master:
        os.makedirs(out_dir, exist_ok=True)

    data_dir = os.path.join(REPO_ROOT, "data", dataset)
    train_x = np.memmap(os.path.join(data_dir, "train_x.bin"), dtype=np.uint16, mode="r")
    train_y = np.memmap(os.path.join(data_dir, "train_y.bin"), dtype=np.uint16, mode="r")
    val_x = np.memmap(os.path.join(data_dir, "val_x.bin"), dtype=np.uint16, mode="r")
    val_y = np.memmap(os.path.join(data_dir, "val_y.bin"), dtype=np.uint16, mode="r")

    def get_batch(split):
        xd, yd = (train_x, train_y) if split == "train" else (val_x, val_y)
        ix = torch.randint(len(xd) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(xd[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(yd[i:i + block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    model = ToyVanilla(config_C).to(device)
    raw_model = model
    if is_ddp:
        ddp_kwargs = {"device_ids": [int(os.environ.get("LOCAL_RANK", rank))]} \
            if torch.cuda.is_available() else {}
        # broadcast_buffers=False: the only buffer is the constant sinusoidal
        # pos_enc, identical on every rank, so there is nothing to sync. Leaving
        # it on would make each forward() issue a broadcast collective, which
        # desyncs the ranks during rank-0-only evaluation.
        model = DDP(model, broadcast_buffers=False, **ddp_kwargs)
        raw_model = model.module

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  betas=(beta1, beta2), weight_decay=weight_decay)

    def get_lr(it):
        if not decay_lr:
            return learning_rate
        if it < warmup_iters:
            return learning_rate * (it + 1) / warmup_iters
        if it > max_iters:
            return min_lr
        ratio = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    @torch.no_grad()
    def eval_val():
        # Evaluate on the UNWRAPPED model: rank 0 runs this alone, so touching
        # the DDP wrapper here would post collectives no other rank matches.
        raw_model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch("val")
            _, loss = raw_model(X, Y)
            losses[k] = loss.item()
        raw_model.train()
        return losses.mean().item()

    ckpt_iters = {name: round(frac * max_iters) for name, frac in ckpt_fracs.items()}
    iter_to_tag = {it: name for name, it in ckpt_iters.items()}
    if is_master:
        print("checkpoint schedule (iter -> tag):", ckpt_iters)

    def save_ckpt(tag, it):
        path = os.path.join(out_dir, f"ckpt_{tag}.pt")
        torch.save({"model": raw_model.state_dict(), "iter_num": it,
                    "tag": tag, "config": config_C.__dict__}, path)
        print(f"  saved checkpoint {path} (iter {it})")

    log_path = os.path.join(out_dir, "loss_log.csv")
    log_rows = []

    model.train()
    t0 = time.time()
    X, Y = get_batch("train")
    for it in range(max_iters + 1):
        for g in optimizer.param_groups:
            g["lr"] = get_lr(it)

        if is_master and it in iter_to_tag:
            save_ckpt(iter_to_tag[it], it)

        if it % eval_interval == 0 and is_master:
            vloss = eval_val()
            print(f"iter {it}: val loss {vloss:.4f}  (lr {get_lr(it):.2e}, {time.time()-t0:.1f}s)")

        if it == max_iters:
            break

        _, loss = model(X, Y)
        X, Y = get_batch("train")          # prefetch next
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if it % log_interval == 0 and is_master:
            lossf = loss.item()
            log_rows.append((it, lossf, get_lr(it)))
            print(f"iter {it}: train loss {lossf:.4f}")

    if is_master:
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "train_loss", "lr"])
            w.writerows(log_rows)
        print("wrote", log_path)

        if log_rows:
            its = [r[0] for r in log_rows]
            ls = [r[1] for r in log_rows]
            plt.figure(figsize=(7, 4.5))
            plt.plot(its, ls, lw=1.2)
            plt.xlabel("iteration"); plt.ylabel("train loss (cross-entropy)")
            plt.title("vanilla_transformer training loss")
            plt.grid(alpha=0.3); plt.tight_layout()
            fig_path = os.path.join(out_dir, "loss_curve.png")
            plt.savefig(fig_path, dpi=150); plt.close()
            print("wrote", fig_path)

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
