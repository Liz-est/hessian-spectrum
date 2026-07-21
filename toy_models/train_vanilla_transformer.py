"""
Train vanilla_transformer (single-layer vanilla decoder) on the synthetic bigram data,
log the loss curve, and checkpoint at the fractions of training named in the
experiment config's ckpt_fracs, so the Hessian analysis (analyze_vanilla.py)
can be run at each stage.

All settings come from the config package (config/schema.py + presets.py):

    python3 train_vanilla_transformer.py                      # default preset
    python3 train_vanilla_transformer.py imbalance_s1_adamw   # a named preset
    python3 train_vanilla_transformer.py --train.max_iters=40 # quick smoke test
    python3 train_vanilla_transformer.py --optim.name=adamw --lr.learning_rate=3e-4
    torchrun --standalone --nproc_per_node=8 train_vanilla_transformer.py   # 8-GPU DDP

The dataset name lives in cfg.data.dataset and resolves to
    <repo-root>/data/<dataset>/
(token ids 0..vocab-1, matching model C's vocab).

Under DDP the per-GPU batch is cfg.data.batch_size; the effective batch is
batch_size * world_size. Rank 0 evaluates, logs, and writes checkpoints.
"""

import os
import sys
import time
import csv
from dataclasses import asdict

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

from vanilla_model import ToyVanilla
import config as cfgmod

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# ----------------------------------------------------------------------------
# config: select a preset (optional bare arg) + --group.key=value overrides.
# All experiment settings now live in the config package, not module globals.
# ----------------------------------------------------------------------------
cfg = cfgmod.apply_overrides(cfgmod.load(), sys.argv[1:])

model_cfg = cfg.to_model_config()
dataset = cfg.data.dataset
batch_size = cfg.data.batch_size
block_size = model_cfg.block_size
max_iters = cfg.train.max_iters
eval_interval = cfg.train.eval_interval
eval_iters = cfg.train.eval_iters
log_interval = cfg.train.log_interval
seed = cfg.train.seed
grad_clip = cfg.optim.grad_clip
out_dir = os.path.join(HERE, "runs", cfg.train.run_name)


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
        print(f"experiment: {cfg.name}  optim={cfg.optim.name}  "
              f"lr={cfg.lr.learning_rate}({cfg.lr.scheduler})  "
              f"max_iters={max_iters}  bs={batch_size}x{world}")

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

    model = ToyVanilla(model_cfg).to(device)
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

    # optimizer + LR schedule are built from the config (name-dispatched),
    # so switching sgd<->adamw or cosine<->constant is a config change only.
    optimizer = cfgmod.build_optimizer(model.parameters(), cfg.optim)
    get_lr = cfgmod.make_lr_fn(cfg.lr, max_iters)

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

    ckpt_iters = cfg.ckpt_iters()   # tag -> iter, with a collision guard
    iter_to_tag = {it: name for name, it in ckpt_iters.items()}
    if is_master:
        print("checkpoint schedule (iter -> tag):", ckpt_iters)

    def save_ckpt(tag, it):
        path = os.path.join(out_dir, f"ckpt_{tag}.pt")
        torch.save({"model": raw_model.state_dict(), "iter_num": it,
                    "tag": tag, "config": model_cfg.__dict__,
                    "experiment": asdict(cfg)}, path)
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
