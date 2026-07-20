"""
Train a vanilla decoder-only Transformer (vanilla_model.ToyVanilla) on the
FineWeb-10B corpus, log the loss curve, and checkpoint at 0% / 10% / 50% / 100%
of training so the Hessian analysis can be run at each stage.

Run from the toy_models/ directory:
    python3 train_vanilla_transformer_fineweb10B.py                 # single process (CPU or 1 GPU)
    python3 train_vanilla_transformer_fineweb10B.py --max_iters=40  # quick smoke test
    torchrun --standalone --nproc_per_node=8 train_vanilla_transformer_fineweb10B.py   # 8-GPU DDP

Data
----
FineWeb-10B lives at <repo-root>/data/fineweb10B/ in the modded-nanoGPT shard
format: each *.bin file is a 1024-byte header (256 int32; magic 20240520,
version 1, ntok) followed by `ntok` uint16 GPT-2 BPE token ids. It is a SINGLE
token stream, so targets are just the inputs shifted by one (next-token
prediction) -- unlike the dual-stream synth data used by train_vanilla_transformer.py.

    fineweb_train_0000{01..20}.bin   (~100M tokens each)
    fineweb_val_000000.bin

Model
-----
Uses vanilla_model.ToyVanillaConfig with the GPT-2 vocab (50304, padded to a
multiple of 64) and a longer context. The default is still the single-layer
vanilla_transformer architecture (d=192) so the Hessian analysis stays tractable; bump
n_layer / n_embd / n_head from the CLI if you want a stronger model.

Under DDP the per-GPU batch is `batch_size`; the effective batch is
batch_size * world_size. Rank 0 evaluates, logs, and writes checkpoints.
"""

import os
import sys
import glob
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

from vanilla_model import ToyVanilla, ToyVanillaConfig

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# ----------------------------------------------------------------------------
# config (override any of these from the CLI as --key=value)
# ----------------------------------------------------------------------------
dataset = "fineweb10B"
# ---- model architecture (defaults = single-layer vanilla_transformer, GPT-2 vocab) --
vocab_size = 50304            # GPT-2 BPE (50257) padded up to a multiple of 64
n_embd = 192
n_head = 6
head_dim = 32                 # n_head * head_dim must equal n_embd
n_ffn = 1024
n_layer = 1
block_size = 1024             # context length (sequence length)
# ---- optimisation ----------------------------------------------------------
batch_size = 32               # per-GPU micro-batch
max_iters = 20000             # 100% of training
warmup_iters = 400
learning_rate = 6e-4
min_lr = 6e-5
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0
decay_lr = True
eval_interval = 500
eval_iters = 100
log_interval = 20
seed = 1337
out_dir = os.path.join(HERE, "runs", "vanilla_fineweb10B")

# checkpoints at these fractions of training (init / 10% / 50% / 100%)
ckpt_fracs = {"init": 0.0, "p10": 0.10, "p50": 0.50, "p100": 1.0}

# ---- minimal CLI override (--key=value) ------------------------------------
for arg in sys.argv[1:]:
    assert arg.startswith("--") and "=" in arg, f"bad arg {arg}"
    key, val = arg[2:].split("=", 1)
    assert key in globals(), f"unknown config key {key}"
    cur = globals()[key]
    globals()[key] = type(cur)(val) if not isinstance(cur, bool) else (val == "True")

# GPT-2 shard header: 256 int32 = 1024 bytes, then uint16 tokens.
HEADER_BYTES = 256 * 4


def load_shards(pattern):
    """Memmap every shard matching `pattern` (uint16 token stream, header skipped)."""
    paths = sorted(glob.glob(pattern))
    assert paths, f"no shards found for {pattern}"
    shards = []
    for p in paths:
        # sanity-check the modded-nanoGPT header on the first shard
        header = np.fromfile(p, dtype=np.int32, count=256)
        assert header[0] == 20240520, f"bad magic in {p}: {header[0]}"
        ntok = int(header[2])
        toks = np.memmap(p, dtype=np.uint16, mode="r", offset=HEADER_BYTES)
        assert len(toks) >= ntok, f"{p}: {len(toks)} < declared {ntok}"
        shards.append(toks[:ntok])
    return shards


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
    assert n_head * head_dim == n_embd, \
        f"n_head*head_dim ({n_head*head_dim}) != n_embd ({n_embd})"

    rank, world, device, is_ddp = setup_dist()
    is_master = rank == 0
    # different seed per rank so each pulls different batches
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    if is_master:
        os.makedirs(out_dir, exist_ok=True)

    data_dir = os.path.join(REPO_ROOT, "data", dataset)
    train_shards = load_shards(os.path.join(data_dir, "fineweb_train_*.bin"))
    val_shards = load_shards(os.path.join(data_dir, "fineweb_val_*.bin"))
    if is_master:
        n_train_tok = sum(len(s) for s in train_shards)
        print(f"train: {len(train_shards)} shards, {n_train_tok/1e9:.3f}B tokens; "
              f"val: {len(val_shards)} shard(s)")

    rng = np.random.default_rng(seed + rank)

    def get_batch(split):
        shards = train_shards if split == "train" else val_shards
        # pick one shard for the whole micro-batch, then random windows in it
        shard = shards[rng.integers(len(shards))]
        # need block_size+1 tokens per sample (inputs + shifted targets)
        ix = rng.integers(0, len(shard) - block_size - 1, size=batch_size)
        x = torch.from_numpy(
            np.stack([shard[i:i + block_size].astype(np.int64) for i in ix]))
        y = torch.from_numpy(
            np.stack([shard[i + 1:i + 1 + block_size].astype(np.int64) for i in ix]))
        return x.to(device), y.to(device)

    config = ToyVanillaConfig(
        vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, head_dim=head_dim,
        n_ffn=n_ffn, n_layer=n_layer, block_size=block_size,
    )
    model = ToyVanilla(config).to(device)
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
                    "tag": tag, "config": config.__dict__}, path)
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
            plt.title("vanilla transformer on FineWeb-10B training loss")
            plt.grid(alpha=0.3); plt.tight_layout()
            fig_path = os.path.join(out_dir, "loss_curve.png")
            plt.savefig(fig_path, dpi=150); plt.close()
            print("wrote", fig_path)

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
