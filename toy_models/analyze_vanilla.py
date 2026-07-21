"""
Analyze trained vanilla_transformer checkpoints and produce every requested figure.

Blocking (per the experiment spec):
  * Q, K (attn.wq/wk)              -> per attention head      (6 units each)
  * V, attn.proj, mlp.fc, mlp.proj -> per output neuron
  * embedding, lm_head             -> per token

Outputs (RUN = cfg.analyze.files_name, set by the experiment preset):
  1. loss curve                    -> runs/<RUN>/loss_curve.png (train_vanilla_transformer.py)
  2. Hessian spectrum (ESD)        -> files/<RUN>/<tag>/spectrum_<layer>.png
  3. per-token / last-layer hetero -> files/<RUN>/<tag>/hetero_<layer>_{skl,js}.png
  4. per-head / per-neuron hetero  -> files/<RUN>/<tag>/hetero_<layer>_{skl,js}.png
  5. cross-LAYER hetero heatmap    -> files/<RUN>/<tag>/hetero_layers_{skl,js}.png
     (pairwise distance between the pooled spectra of all analyzed layers)
  6. hetero-vs-epoch evolution     -> files/<RUN>/evolution_{skl,js}.png
     (one x tick per checkpoint tag in cfg.train.ckpt_fracs, one line per layer)
  7. cross-layer hetero evolution  -> files/<RUN>/evolution_layers_{skl,js}.png

Runs on CPU (single process) or on 8 GPUs (torchrun). Under torchrun the
(checkpoint, layer) work items are sharded across ranks; every rank writes its
own eigs_*.npy / hetero_*.npy, then rank 0 renders all figures after a barrier.
Settings come from the config package -- select a preset / override fields the
same way as the trainer, e.g.:

    cd toy_models
    python3 analyze_vanilla.py                                   # default preset
    python3 analyze_vanilla.py imbalance_s1_adamw                # a named preset
    python3 analyze_vanilla.py --analyze.max_classes=1024        # full-vocab lm_head
    torchrun --standalone --nproc_per_node=8 analyze_vanilla.py  # 8-GPU sharded
"""

import os
import sys
import json

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import torch.distributed as dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vanilla_model import ToyVanilla
import config as cfgmod
from hessian_toy import (NeuronHessian, analyze_layer, default_layer_spec,
                         spectra_to_prob, common_log_edges,
                         cross_layer_matrices, hetero_mean)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# ---- config: same preset selection + --group.key=value overrides as training.
# Reading from the same ExperimentConfig means the checkpoint tags analysed here
# (cfg.train.ckpt_fracs) always match the ones the trainer wrote -- no separate
# TAGS list to keep in sync.
cfg = cfgmod.apply_overrides(cfgmod.load(), sys.argv[1:])

model_cfg = cfg.to_model_config()
dataset = cfg.data.dataset
data_format = cfg.data.format
run_dir = os.path.join(HERE, "runs", cfg.train.run_name)     # where ckpt_*.pt live
out_dir = os.path.join(HERE, "files", cfg.analyze.files_name)  # eigs/hetero npy + figures
batch_size = cfg.analyze.batch_size
n_batches = cfg.analyze.n_batches
max_classes = cfg.analyze.max_classes
max_tokens = cfg.analyze.max_tokens
num_bins = cfg.analyze.num_bins
seed = cfg.analyze.seed

# checkpoint tags in training order, paired with the fraction they mark
TAGS = sorted(cfg.train.ckpt_fracs.items(), key=lambda kv: kv[1])

LAYER_SPEC = default_layer_spec(model_cfg.n_head, model_cfg.head_dim,
                                n_layer=model_cfg.n_layer,
                                block_type=model_cfg.block_type)
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


def make_get_batch(block_size, device):
    data_dir = os.path.join(REPO_ROOT, "data", dataset)
    if data_format == "nanogpt_shards":
        # fineweb-style single token stream; targets are inputs shifted by one.
        import glob
        HEADER_BYTES = 256 * 4
        paths = sorted(glob.glob(os.path.join(data_dir, "fineweb_train_*.bin")))
        assert paths, f"no shards found in {data_dir}"
        shards = []
        for p in paths:
            header = np.fromfile(p, dtype=np.int32, count=256)
            assert header[0] == 20240520, f"bad magic in {p}: {header[0]}"
            ntok = int(header[2])
            toks = np.memmap(p, dtype=np.uint16, mode="r", offset=HEADER_BYTES)
            shards.append(toks[:ntok])
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

    # dual-stream synth bigram data: x and y stored in separate .bin files.
    xd = np.memmap(os.path.join(data_dir, "train_x.bin"), dtype=np.uint16, mode="r")
    yd = np.memmap(os.path.join(data_dir, "train_y.bin"), dtype=np.uint16, mode="r")
    g = torch.Generator().manual_seed(seed)

    def get_batch():
        ix = torch.randint(len(xd) - block_size, (batch_size,), generator=g)
        x = torch.stack([torch.from_numpy(xd[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(yd[i:i + block_size].astype(np.int64)) for i in ix])
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


def plot_heatmap(save_dir, name, D, metric, title, labels=None):
    mask = np.triu(np.ones_like(D, dtype=bool), k=1)
    Dm = np.ma.array(D, mask=mask)
    vmax = float(np.sqrt(np.log(2.0))) if metric == "js" else None
    plt.figure(figsize=(6.5, 5.2))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    im = plt.imshow(Dm, cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
    label = "Symmetric KL" if metric == "skl" else "JS distance"
    plt.colorbar(im, label=label, fraction=0.046, pad=0.04)
    plt.title(title)
    if labels is not None:
        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=45, ha="right", fontsize=8)
        plt.yticks(ticks, labels, fontsize=8)
    else:
        plt.xlabel("unit index"); plt.ylabel("unit index")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"hetero_{name}_{metric}.png"), dpi=150)
    plt.close()


def render_all_figs(all_results):
    layer_means = {}   # tag -> {metric: lower-triangle mean of the cross-layer matrix}
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

        # cross-LAYER hetero: one pooled spectrum per layer, pairwise distances
        res = cross_layer_matrices(save_dir, LAYER_NAMES, num_bins=num_bins)
        if res is not None:
            layers, mats = res
            layer_means[tag] = {}
            for metric in ("skl", "js"):
                plot_heatmap(save_dir, "layers", mats[metric], metric,
                             f"cross-layer hetero ({metric.upper()}, {tag})",
                             labels=layers)
                layer_means[tag][metric] = hetero_mean(mats[metric])

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

    # evolution of the CROSS-LAYER hetero: mean pairwise distance between layers
    for metric in ("skl", "js"):
        xs = [frac * 100 for tag, frac in TAGS if tag in layer_means]
        ys = [layer_means[tag][metric] for tag, _ in TAGS if tag in layer_means]
        if not xs:
            continue
        plt.figure(figsize=(8, 5.5))
        plt.plot(xs, ys, marker="o", color="darkslateblue")
        plt.xlabel("training progress (% of iters)")
        ylab = "mean Symmetric KL" if metric == "skl" else "mean JS distance"
        plt.ylabel(ylab + " (lower-triangle, layer pairs)")
        plt.title(f"Cross-layer Hessian heterogeneity vs training ({metric.upper()})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, f"evolution_layers_{metric}.png")
        plt.savefig(path, dpi=150); plt.close()
        print("wrote", path)


def main():
    rank, world, device, is_ddp = setup_dist()
    is_master = rank == 0
    if is_master:
        os.makedirs(out_dir, exist_ok=True)
    if is_ddp:
        dist.barrier()

    get_batch = make_get_batch(model_cfg.block_size, device)

    # work items: (tag, layer). Shard strided across ranks.
    work = [(tag, item) for (tag, _) in TAGS for item in LAYER_SPEC]
    my_work = work[rank::world]

    # cache one loaded model per tag (avoid reloading per layer)
    model_cache = {}

    def get_model(tag):
        if tag not in model_cache:
            ckpt_path = os.path.join(run_dir, f"ckpt_{tag}.pt")
            ckpt = torch.load(ckpt_path, map_location=device)
            m = ToyVanilla(model_cfg).to(device)
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
                      model_cfg.n_head, model_cfg.head_dim,
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
            print("no checkpoints found; run train_vanilla_transformer.py first.")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
