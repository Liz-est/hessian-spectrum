"""
FULL-BATCH training of the toy models on the synthetic dual-stream data:
every iteration computes the gradient of the loss over the ENTIRE train split
(one deterministic batch), so GD / Adam here are the exact full-batch methods,
not their stochastic variants. Built as a minimal fork of
train_vanilla_transformer.py -- same config package, same checkpoint schedule,
same logs -- with these deliberate differences:

  * no random get_batch: the train split is reshaped once into
    (n_seq, block_size) rows and reused every step. cfg.data.batch_size is
    repurposed as the MICRO-batch size for gradient accumulation, so memory
    stays bounded while the accumulated gradient equals the true full-batch
    gradient exactly (all rows have identical token counts, so the per-row
    losses average with equal weights).
  * eval is the exact loss over the full split (train and val), not a random
    eval_iters-batch estimate.
  * single-process only (no DDP): the full batch fits one device, and DDP
    averaging would just re-implement the accumulation loop.

Intended presets (config/presets.py): fullbatch-mse0-shuffled-2p17-{gd,adam} --
the 0-layer embed+lm_head MSE model without pos_enc (_MODEL_EMBED_HEAD_MSE_POS0)
on the shuffled-marginals data from build_shuffled_dataset.py, whose 2^17-pair
train split is sized to BE the full batch. The 2^14-pair validation split and
the train split are both exact multiples of block_size=128.

    python3 train_fullbatch.py fullbatch-mse0-shuffled-2p17-gd
    python3 train_fullbatch.py fullbatch-mse0-shuffled-2p17-adam
    python3 train_fullbatch.py fullbatch-mse0-shuffled-2p17-gd --train.max_iters=40
    python3 train_fullbatch.py fullbatch-mse0-shuffled-2p17-adam \
        --model.tok_emb_init_std=0.05 --model.lm_head_init_std=0
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vanilla_model import ToyVanilla
import config as cfgmod

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# ----------------------------------------------------------------------------
# config: select a preset (optional bare arg) + --group.key=value overrides.
# ----------------------------------------------------------------------------
cfg = cfgmod.apply_overrides(cfgmod.load(), sys.argv[1:])

model_cfg = cfg.to_model_config()
dataset = cfg.data.dataset
micro_bs = cfg.data.batch_size     # gradient-accumulation micro-batch (rows)
block_size = model_cfg.block_size
max_iters = cfg.train.max_iters
eval_interval = cfg.train.eval_interval
log_interval = cfg.train.log_interval
seed = cfg.train.seed
grad_clip = cfg.optim.grad_clip
out_dir = os.path.join(HERE, "runs", cfg.train.run_name)


def resolve_data_dir(name):
    """<repo-root>/data/<name> like the other trainers, falling back to
    data_construction/data/<name> where build_shuffled_dataset.py writes."""
    for d in (os.path.join(REPO_ROOT, "data", name),
              os.path.join(REPO_ROOT, "data_construction", "data", name)):
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"dataset {name!r} not found under "
                            f"{REPO_ROOT}/data or {REPO_ROOT}/data_construction/data")


def load_split(data_dir, prefix, device):
    """Load <prefix>_x.bin / <prefix>_y.bin as the exact full split.

    Require complete block_size rows instead of silently dropping a tail: the
    dataset generator uses power-of-two split sizes, so the default block size
    divides both splits exactly.
    """
    x = np.fromfile(os.path.join(data_dir, f"{prefix}_x.bin"), dtype=np.uint16)
    y = np.fromfile(os.path.join(data_dir, f"{prefix}_y.bin"), dtype=np.uint16)
    if len(x) != len(y):
        raise ValueError(f"{prefix}: x/y length mismatch ({len(x)} != {len(y)})")
    if len(x) % block_size:
        raise ValueError(
            f"{prefix}: {len(x)} pairs is not divisible by block_size={block_size}; "
            "regenerate the split or choose a compatible block size"
        )
    n_seq = len(x) // block_size
    X = torch.from_numpy(x.astype(np.int64)).view(n_seq, block_size)
    Y = torch.from_numpy(y.astype(np.int64)).view(n_seq, block_size)
    return X.to(device), Y.to(device)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    print(f"experiment: {cfg.name}  optim={cfg.optim.name}  "
          f"lr={cfg.lr.learning_rate}({cfg.lr.scheduler})  "
          f"max_iters={max_iters}  FULL-BATCH (micro_bs={micro_bs})")

    data_dir = resolve_data_dir(dataset)
    train_X, train_Y = load_split(data_dir, "train", device)
    val_X, val_Y = load_split(data_dir, "val", device)
    print(f"full batch: train {tuple(train_X.shape)} rows x block, "
          f"val {tuple(val_X.shape)}  ({data_dir})")

    model = ToyVanilla(model_cfg).to(device)
    if cfg.train.freeze:
        frozen = cfgmod.freeze_submodules(model, cfg.train.freeze)
        desc = ", ".join(f"{k} ({v} params)" for k, v in frozen.items())
        print(f"frozen at init: {desc}")

    optimizer = cfgmod.build_optimizer(model, cfg.optim)
    get_lr = cfgmod.make_lr_fn(cfg.lr, max_iters)

    def full_backward(X, Y):
        """Accumulate the EXACT full-batch mean loss + gradient in micro-batch
        chunks. Every row contributes block_size positions, so weighting each
        chunk's mean loss by its row fraction reproduces the global mean."""
        n = X.size(0)
        total = 0.0
        for i in range(0, n, micro_bs):
            xb, yb = X[i:i + micro_bs], Y[i:i + micro_bs]
            _, loss = model(xb, yb)
            w = xb.size(0) / n
            (loss * w).backward()
            total += loss.item() * w
        return total

    @torch.no_grad()
    def full_loss(X, Y):
        """Exact mean loss over a whole split (no gradient)."""
        model.eval()
        n = X.size(0)
        total = 0.0
        for i in range(0, n, micro_bs):
            xb, yb = X[i:i + micro_bs], Y[i:i + micro_bs]
            _, loss = model(xb, yb)
            total += loss.item() * (xb.size(0) / n)
        model.train()
        return total

    ckpt_iters = cfg.ckpt_iters()   # tag -> iter, with a collision guard
    iter_to_tag = {it: name for name, it in ckpt_iters.items()}
    print("checkpoint schedule (iter -> tag):", ckpt_iters)

    def save_ckpt(tag, it):
        path = os.path.join(out_dir, f"ckpt_{tag}.pt")
        torch.save({"model": model.state_dict(), "iter_num": it,
                    "tag": tag, "config": model_cfg.__dict__,
                    "experiment": asdict(cfg)}, path)
        print(f"  saved checkpoint {path} (iter {it})")

    log_path = os.path.join(out_dir, "loss_log.csv")
    val_log_path = os.path.join(out_dir, "val_loss_log.csv")
    log_rows = []
    val_rows = []

    model.train()
    t0 = time.time()
    for it in range(max_iters + 1):
        for g in optimizer.param_groups:
            g["lr"] = get_lr(it)

        if it in iter_to_tag:
            save_ckpt(iter_to_tag[it], it)

        if it % eval_interval == 0:
            vloss = full_loss(val_X, val_Y)
            val_rows.append((it, vloss))
            print(f"iter {it}: val loss {vloss:.6f}  (lr {get_lr(it):.2e}, {time.time()-t0:.1f}s)")

        if it == max_iters:
            break

        optimizer.zero_grad(set_to_none=True)
        lossf = full_backward(train_X, train_Y)   # exact full-batch gradient
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if it % log_interval == 0:
            log_rows.append((it, lossf, get_lr(it)))
            print(f"iter {it}: train loss {lossf:.6f}")

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "train_loss", "lr"])
        w.writerows(log_rows)
    print("wrote", log_path)

    with open(val_log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "val_loss"])
        w.writerows(val_rows)
    print("wrote", val_log_path)

    if log_rows:
        its = [r[0] for r in log_rows]
        ls = [r[1] for r in log_rows]
        plt.figure(figsize=(7, 4.5))
        plt.plot(its, ls, lw=1.2, label="train (full batch)")
        if val_rows:
            plt.plot([r[0] for r in val_rows], [r[1] for r in val_rows],
                     lw=1.2, marker="o", ms=3, label="val (full)")
        loss_label = "MSE (vs one-hot)" if cfg.model.loss_type == "mse" else "cross-entropy"
        plt.xlabel("iteration"); plt.ylabel(f"loss ({loss_label})")
        plt.title(f"{cfg.train.run_name} loss")
        plt.legend()
        plt.grid(alpha=0.3); plt.tight_layout()
        fig_path = os.path.join(out_dir, "loss_curve.png")
        plt.savefig(fig_path, dpi=150); plt.close()
        print("wrote", fig_path)


if __name__ == "__main__":
    main()
