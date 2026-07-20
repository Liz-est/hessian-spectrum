"""
Analyze trained vanilla-transformer (FineWeb-10B) checkpoints and produce every
requested figure. This is the FineWeb counterpart of analyze_vanilla.py: same Hessian
machinery, but it reads the modded-nanoGPT single-stream token shards and builds
the model config from CLI params (must match the training run) instead of
importing config_C.

Blocking (per the experiment spec):
  * Q, K (attn.wq/wk)              -> per attention head      (n_head units each)
  * V, attn.proj, mlp.fc, mlp.proj -> per output neuron
  * embedding, lm_head             -> per token

Outputs:
  1. loss curve                    -> runs/vanilla_fineweb10B/loss_curve.png (training script)
  2. Hessian spectrum (ESD)        -> files/vanilla_fineweb10B/<tag>/spectrum_<layer>.png
  3. per-token / last-layer hetero -> files/vanilla_fineweb10B/<tag>/hetero_<layer>_{skl,js}.png
  4. per-head / per-neuron hetero  -> files/vanilla_fineweb10B/<tag>/hetero_<layer>_{skl,js}.png
  5. hetero-vs-epoch evolution     -> files/vanilla_fineweb10B/evolution_{skl,js}.png
     (init / 10% / 50% / 100% on the x-axis, one line per analyzed layer)

Runs on CPU (single process) or on 8 GPUs (torchrun). Under torchrun the
(checkpoint, layer) work items are sharded across ranks; every rank writes its
own eigs_*.npy / hetero_*.npy, then rank 0 renders all figures after a barrier.

    cd toy_models
    python3 analyze_vanilla_transformer_fineweb10B.py                                   # single process
    torchrun --standalone --nproc_per_node=8 analyze_vanilla_transformer_fineweb10B.py  # 8-GPU sharded

The model-architecture args below (vocab_size / n_embd / n_head / head_dim /
n_ffn / n_layer / block_size) MUST match the ones used for training, otherwise
the checkpoint state_dict will not load. Override them the same way you did in
train_vanilla_transformer_fineweb10B.py.
"""

import os
import sys
import glob
import json

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import torch.distributed as dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vanilla_model import ToyVanilla, ToyVanillaConfig
from hessian_toy import (NeuronHessian, analyze_layer, default_layer_spec,
                         spectra_to_prob, common_log_edges)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# ---- config (CLI overridable via --key=value) ----
dataset = "fineweb10B"
run_dir = os.path.join(HERE, "runs", "vanilla_fineweb10B")
out_dir = os.path.join(HERE, "files", "vanilla_fineweb10B")
# ---- model architecture (MUST match the training run) ----
vocab_size = 50304
n_embd = 192
n_head = 6
head_dim = 32
n_ffn = 1024
n_layer = 1
block_size = 1024
# ---- curvature-estimation params ----
batch_size = 32
n_batches = 20             # curvature batches per layer
max_classes = 256          # per-class blocks for the lm_head (subset for speed)
max_tokens = 256           # per-token blocks for the embedding (subset)
num_bins = 64
seed = 1337

# checkpoint tags in training order, with the fraction of training they mark
TAGS = [("init", 0.0), ("p10", 0.10), ("p50", 0.50), ("p100", 1.0)]

for arg in sys.argv[1:]:
    assert arg.startswith("--") and "=" in arg, f"bad arg {arg}"
    key, val = arg[2:].split("=", 1)
    assert key in globals(), f"unknown key {key}"
    cur = globals()[key]
    globals()[key] = type(cur)(val)

# GPT-2 shard header: 256 int32 = 1024 bytes, then uint16 tokens.
HEADER_BYTES = 256 * 4

CONFIG = ToyVanillaConfig(
    vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, head_dim=head_dim,
    n_ffn=n_ffn, n_layer=n_layer, block_size=block_size,
)

LAYER_SPEC = default_layer_spec(CONFIG.n_head, CONFIG.head_dim)
LAYER_NAMES = [d for (d, _, _) in LAYER_SPEC]


# ----------------------------------------------------------------------------
# distributed setup (no-op when launched as a single process)
# ----------------------------------------------------------------------------
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


def load_shards(pattern):
    """Memmap every shard matching `pattern` (uint16 token stream, header skipped)."""
    paths = sorted(glob.glob(pattern))
    assert paths, f"no shards found for {pattern}"
    shards = []
    for p in paths:
        header = np.fromfile(p, dtype=np.int32, count=256)
        assert header[0] == 20240520, f"bad magic in {p}: {header[0]}"
        ntok = int(header[2])
        toks = np.memmap(p, dtype=np.uint16, mode="r", offset=HEADER_BYTES)
        assert len(toks) >= ntok, f"{p}: {len(toks)} < declared {ntok}"
        shards.append(toks[:ntok])
    return shards


def make_get_batch(block_size, device):
    data_dir = os.path.join(REPO_ROOT, "data", dataset)
    shards = load_shards(os.path.join(data_dir, "fineweb_train_*.bin"))
    rng = np.random.default_rng(seed)

    def get_batch():
        shard = shards[rng.integers(len(shards))]
        ix = rng.integers(0, len(shard) - block_size - 1, size=batch_size)
        x = torch.from_numpy(
            np.stack([shard[i:i + block_size].astype(np.int64) for i in ix]))
        y = torch.from_numpy(
            np.stack([shard[i + 1:i + 1 + block_size].astype(np.int64) for i in ix]))
        return x.to(device), y.to(device)
    return get_batch


# ----------------------------------------------------------------------------
# figures (rank 0 only)
# ----------------------------------------------------------------------------
def plot_spectrum(save_dir, name, eigs, title):
    vals = np.clip(eigs.ravel(), 0.0, None)
    vals = vals[np.isfinite(vals)]
    pos = vals[vals > 0]
    if pos.size == 0:
        return
    edges = np.linspace(0, np.quantile(vals, 0.999) + 1e-12, 80)
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=edges, color="steelblue", alpha=0.8)
    plt.xlabel("eigenvalue"); plt.ylabel("count"); plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"spectrum_{name}.png"), dpi=150)
    plt.close()

    logedges = np.linspace(np.log10(pos.min() + 1e-12),
                           np.log10(pos.max() + 1e-12), 80)
    plt.figure(figsize=(6, 4))
    plt.hist(np.log10(pos), bins=logedges, color="indianred", alpha=0.8)
    plt.xlabel(r"$\log_{10}(\lambda)$"); plt.ylabel("count")
    plt.title(title + " (log-eigenvalue)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"spectrum_{name}_log.png"), dpi=150)
    plt.close()


def plot_heatmap(save_dir, name, D, metric, title):
    mask = np.triu(np.ones_like(D, dtype=bool), k=1)
    Dm = np.ma.array(D, mask=mask)
    vmax = float(np.sqrt(np.log(2.0))) if metric == "js" else None
    plt.figure(figsize=(6.5, 5.2))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    im = plt.imshow(Dm, cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
    label = "Symmetric KL" if metric == "skl" else "JS distance"
    plt.colorbar(im, label=label, fraction=0.046, pad=0.04)
    plt.title(title); plt.xlabel("unit index"); plt.ylabel("unit index")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"hetero_{name}_{metric}.png"), dpi=150)
    plt.close()


def render_all_figs(all_results):
    for tag, _ in TAGS:
        save_dir = os.path.join(out_dir, tag)
        if not os.path.isdir(save_dir):
            continue
        for disp in LAYER_NAMES:
            ef = os.path.join(save_dir, f"eigs_{disp}.npy")
            if not os.path.exists(ef):
                continue
            eigs = np.load(ef)
            plot_spectrum(save_dir, disp, eigs, f"{disp} ESD ({tag})")
            for metric in ("skl", "js"):
                D = np.load(os.path.join(save_dir, f"hetero_{disp}_{metric}.npy"))
                plot_heatmap(save_dir, disp, D, metric,
                             f"{disp} hetero ({metric.upper()}, {tag})")

    # evolution: one line per layer, x = training %
    for metric in ("skl", "js"):
        plt.figure(figsize=(8, 5.5))
        for disp in LAYER_NAMES:
            xs, ys = [], []
            for tag, frac in TAGS:
                info = all_results.get(tag, {}).get(disp)
                if info is not None:
                    xs.append(frac * 100)
                    ys.append(info[f"{metric}_mean"])
            if xs:
                plt.plot(xs, ys, marker="o", label=disp)
        plt.xlabel("training progress (% of iters)")
        ylab = "mean Symmetric KL" if metric == "skl" else "mean JS distance"
        plt.ylabel(ylab + " (lower-triangle)")
        plt.title(f"Hessian heterogeneity vs training ({metric.upper()})")
        plt.grid(alpha=0.3); plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        path = os.path.join(out_dir, f"evolution_{metric}.png")
        plt.savefig(path, dpi=150); plt.close()
        print("wrote", path)


def main():
    rank, world, device, is_ddp = setup_dist()
    is_master = rank == 0
    if is_master:
        os.makedirs(out_dir, exist_ok=True)
    if is_ddp:
        dist.barrier()

    get_batch = make_get_batch(CONFIG.block_size, device)

    # work items: (tag, layer). Shard strided across ranks.
    work = [(tag, item) for (tag, _) in TAGS for item in LAYER_SPEC]
    my_work = work[rank::world]

    # cache one loaded model per tag (avoid reloading per layer)
    model_cache = {}

    def get_model(tag):
        if tag not in model_cache:
            ckpt_path = os.path.join(run_dir, f"ckpt_{tag}.pt")
            ckpt = torch.load(ckpt_path, map_location=device)
            m = ToyVanilla(CONFIG).to(device)
            m.load_state_dict(ckpt["model"])
            model_cache[tag] = m
        return model_cache[tag]

    for tag, (disp, kind, kwargs) in my_work:
        ckpt_path = os.path.join(run_dir, f"ckpt_{tag}.pt")
        if not os.path.exists(ckpt_path):
            if is_master:
                print(f"[skip] checkpoint missing: {ckpt_path}")
            continue
        model = get_model(tag)
        nh = NeuronHessian(model, get_batch, n_batches=n_batches, device=device)
        print(f"[rank {rank}] {tag}/{disp} ({kind}) ...", flush=True)
        analyze_layer(nh, out_dir, tag, disp, kind, kwargs,
                      CONFIG.n_head, CONFIG.head_dim,
                      max_classes=max_classes, max_tokens=max_tokens,
                      num_bins=num_bins, device=device)

    if is_ddp:
        dist.barrier()

    if is_master:
        # gather every per-layer summary that got written and render figures
        all_results = {}
        for tag, _ in TAGS:
            save_dir = os.path.join(out_dir, tag)
            for disp in LAYER_NAMES:
                sp = os.path.join(save_dir, f"summary_{disp}.json")
                if os.path.exists(sp):
                    with open(sp) as f:
                        all_results.setdefault(tag, {})[disp] = json.load(f)
        if all_results:
            render_all_figs(all_results)
            with open(os.path.join(out_dir, "all_summary.json"), "w") as f:
                json.dump(all_results, f, indent=2)
            print("wrote", os.path.join(out_dir, "all_summary.json"))
        else:
            print("no checkpoints found; run train_vanilla_transformer_fineweb10B.py first.")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
