"""
Full-parameter Hessian ESD via stochastic Lanczos quadrature (SLQ).

Unlike analyze_vanilla.py (per-unit block spectra, block-diagonal view), this
computes the spectral density of the FULL Hessian of the loss w.r.t. the whole
parameter vector -- cross-block curvature included -- following the method of
language_models/hessian_spectrum.py (paper arXiv 2402.16788), with two
accuracy upgrades that the tiny toy models afford:

  * fp64 end-to-end: the model is cast to double and the HVP / Lanczos
    recursion run in float64 (a 3.1M-param model is ~25 MB in fp64).
  * FULL reorthogonalization: all m Lanczos vectors are kept on GPU
    (m=100 x 3.1M x 8B ~ 2.5 GB) and each new vector is re-orthogonalized
    against the whole basis, killing the ghost-eigenvalue problem of plain
    three-term Lanczos.

The Hessian operator is the loss averaged over a FIXED set of
cfg.analyze.slq_n_batches batches (same seed everywhere), so every probe /
rank / checkpoint sees the same deterministic matrix.

Work items are (checkpoint tag, probe k); under torchrun they are sharded
strided across ranks -- probes are independent Lanczos runs, so there is no
per-step communication at all. Each item writes
    files/<FILES>/slq/<tag>/ritz_v<k>.npz     (Ritz values + weights)
and rank 0 then renders, per tag and overall:
    files/<FILES>/slq/spectrum_full_<tag>.png        (density, semilogy)
    files/<FILES>/slq/spectrum_full_evolution.png    (all tags overlaid)
    files/<FILES>/slq/summary_slq.json               (lambda_max/min, trace)

A ritz_v<k>.npz that already exists is skipped, so re-running resumes.

Usage (same preset/override CLI as the other tools):
    python analyze_full_spectrum.py layer5-imbalance-s1-sgd \
        --train.run_name=grid-search-1/l5-imb-sgd-lr0p25-wd1em05 \
        --analyze.files_name=grid-search-1-best-lr0p25-wd1em05
    torchrun --standalone --nproc_per_node=8 analyze_full_spectrum.py ...
"""

import json
import math
import os
import sys
import time
from datetime import timedelta

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import torch.distributed as dist
from torch.nn.attention import sdpa_kernel, SDPBackend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vanilla_model import ToyVanilla
import config as cfgmod

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

cfg = cfgmod.apply_overrides(cfgmod.load(), sys.argv[1:])

model_cfg = cfg.to_model_config()
run_dir = os.path.join(HERE, "runs", cfg.train.run_name)
out_dir = os.path.join(HERE, "files", cfg.analyze.files_name, "slq")

M = cfg.analyze.slq_m                  # Lanczos steps per probe
NUM_V = cfg.analyze.slq_num_v          # probes per checkpoint
N_BATCHES = cfg.analyze.slq_n_batches  # fixed batches defining the operator
SIGMA2 = cfg.analyze.slq_sigma2        # gaussian broadening variance
BATCH_SIZE = cfg.analyze.batch_size
SEED = cfg.analyze.seed
DTYPE = torch.float64 if cfg.analyze.slq_dtype == "fp64" else torch.float32

TAGS = sorted(cfg.train.ckpt_fracs.items(), key=lambda kv: kv[1])


def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(hours=4))
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            local = int(os.environ.get("LOCAL_RANK", rank))
            torch.cuda.set_device(local)
            device = f"cuda:{local}"
        else:
            device = "cpu"
        return rank, world, device, True
    return 0, 1, "cuda:0" if torch.cuda.is_available() else "cpu", False


def load_fixed_batches(device):
    """The FIXED batch set that defines the Hessian operator.

    Same seed -> identical batches on every rank / probe / checkpoint, so all
    Lanczos runs see one deterministic matrix. Total tokens are tiny
    (n_batches * batch_size * block_size ints), so everything sits on GPU.
    """
    data_dir = os.path.join(REPO_ROOT, "data", cfg.data.dataset)
    bs = model_cfg.block_size
    if cfg.data.format == "nanogpt_shards":
        import glob
        HEADER_BYTES = 256 * 4
        paths = sorted(glob.glob(os.path.join(data_dir, "fineweb_train_*.bin")))
        assert paths, f"no shards found in {data_dir}"
        header = np.fromfile(paths[0], dtype=np.int32, count=256)
        assert header[0] == 20240520, f"bad magic in {paths[0]}"
        toks = np.memmap(paths[0], dtype=np.uint16, mode="r",
                         offset=HEADER_BYTES)[:int(header[2])]
        rng = np.random.default_rng(SEED)
        batches = []
        for _ in range(N_BATCHES):
            ix = rng.integers(0, len(toks) - bs - 1, size=BATCH_SIZE)
            x = np.stack([toks[i:i + bs].astype(np.int64) for i in ix])
            y = np.stack([toks[i + 1:i + 1 + bs].astype(np.int64) for i in ix])
            batches.append((torch.from_numpy(x).to(device),
                            torch.from_numpy(y).to(device)))
        return batches

    xd = np.memmap(os.path.join(data_dir, "train_x.bin"), dtype=np.uint16, mode="r")
    yd = np.memmap(os.path.join(data_dir, "train_y.bin"), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(SEED)
    batches = []
    for _ in range(N_BATCHES):
        ix = rng.integers(0, len(xd) - bs, size=BATCH_SIZE)
        x = np.stack([xd[i:i + bs].astype(np.int64) for i in ix])
        y = np.stack([yd[i:i + bs].astype(np.int64) for i in ix])
        batches.append((torch.from_numpy(x).to(device),
                        torch.from_numpy(y).to(device)))
    return batches


class FullHessian:
    """HVP of the loss averaged over the fixed batch set, at fixed params."""

    def __init__(self, model, batches):
        self.model = model
        self.batches = batches
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.dim = sum(p.numel() for p in self.params)

    def _flat(self, tensors):
        return torch.cat([t.reshape(-1) for t in tensors])

    def hvp(self, v_flat):
        """H @ v, averaged over the fixed batches. v_flat: (dim,) DTYPE."""
        vs, off = [], 0
        for p in self.params:
            vs.append(v_flat[off:off + p.numel()].view_as(p))
            off += p.numel()
        out = torch.zeros_like(v_flat)
        # flash / mem-efficient SDPA kernels have no double-backward: force math
        with sdpa_kernel(SDPBackend.MATH):
            for X, Y in self.batches:
                _, loss = self.model(X, Y)
                g = torch.autograd.grad(loss, self.params, create_graph=True)
                gv = sum((gi * vi).sum() for gi, vi in zip(g, vs))
                h = torch.autograd.grad(gv, self.params)
                out += self._flat([t.detach() for t in h])
        return out / len(self.batches)


def lanczos_full_reorth(hess, k, device):
    """m-step Lanczos with FULL reorthogonalization; probe seeded by k.

    Returns (ritz_values, ritz_weights): eigen-decomposition of the m x m
    tridiagonal T, weights = squared first components (SLQ quadrature weights).
    """
    D = hess.dim
    gen = torch.Generator(device="cpu").manual_seed(SEED + 7919 * (k + 1))
    v = torch.randn(D, generator=gen, dtype=DTYPE).to(device)
    v /= v.norm()

    V = torch.zeros(M, D, dtype=DTYPE, device=device)  # full basis for reorth
    T = np.zeros((M, M), dtype=np.float64)
    V[0] = v

    w = hess.hvp(v)
    alpha = torch.dot(w, v)
    w -= alpha * v
    T[0, 0] = alpha.item()

    for j in range(1, M):
        beta = w.norm()
        if beta < 1e-10:
            # invariant subspace found: T[:j,:j] is exact, truncate
            T = T[:j, :j]
            V = V[:j]
            break
        v = w / beta
        # full reorthogonalization (twice is enough - Parlett)
        for _ in range(2):
            v -= V[:j].T @ (V[:j] @ v)
        v /= v.norm()
        V[j] = v

        w = hess.hvp(v)
        alpha = torch.dot(w, v)
        w = w - alpha * v - beta * V[j - 1]
        T[j, j] = alpha.item()
        T[j - 1, j] = T[j, j - 1] = beta.item()

    evals, U = np.linalg.eigh(T)
    return evals, U[0] ** 2


# ----------------------------------------------------------------------------
# density reconstruction + figures (rank 0)
# ----------------------------------------------------------------------------
def gaussian_density(grid, values, weights, sigma2):
    """sum_j w_j * N(grid; theta_j, sigma2)  (vectorized over the grid)."""
    coeff = 1.0 / math.sqrt(2 * math.pi * sigma2)
    d = grid[:, None] - values[None, :]
    return coeff * (np.exp(-d * d / (2 * sigma2)) * weights[None, :]).sum(axis=1)


def tag_density(ritz):
    """Average the per-probe quadrature densities on a common grid."""
    vmin = min(v.min() for v, _ in ritz)
    vmax = max(v.max() for v, _ in ritz)
    pad = 0.05 * (vmax - vmin) + 10 * math.sqrt(SIGMA2)
    grid = np.linspace(vmin - pad, vmax + pad, 50000)
    dens = np.mean([gaussian_density(grid, v, w, SIGMA2) for v, w in ritz], axis=0)
    dens /= dens.sum() * (grid[1] - grid[0])
    return grid, dens


def render(all_ritz):
    summary = {}
    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(8, 5))
    done_tags = [t for t, _ in TAGS if t in all_ritz]
    for i, tag in enumerate(done_tags):
        ritz = all_ritz[tag]
        grid, dens = tag_density(ritz)
        np.savez(os.path.join(out_dir, f"density_{tag}.npz"), grid=grid, density=dens)
        summary[tag] = {
            "lambda_max": float(max(v.max() for v, _ in ritz)),
            "lambda_min": float(min(v.min() for v, _ in ritz)),
            "trace_over_dim": float(np.mean([(v * w).sum() for v, w in ritz])),
            "num_v": len(ritz), "m": M, "n_batches": N_BATCHES,
            "batch_size": BATCH_SIZE, "sigma2": SIGMA2,
            "dtype": str(DTYPE).split(".")[-1],
        }
        # per-tag semilogy figure (paper style)
        plt.figure(figsize=(7, 4.5))
        plt.semilogy(grid, np.maximum(dens, 1e-12), lw=1.0)
        plt.ylim([1e-10, 1e3])
        plt.xlabel("eigenvalue")
        plt.ylabel("density (log)")
        plt.title(f"full-parameter Hessian ESD, SLQ ({tag})\n"
                  f"m={M}, num_v={len(ritz)}, full reorth, {summary[tag]['dtype']}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"spectrum_full_{tag}.png"), dpi=150)
        plt.close()

        plt.figure(1)
        plt.semilogy(grid, np.maximum(dens, 1e-12), lw=1.2, label=tag,
                     color=cmap(i / max(len(done_tags) - 1, 1)))

    plt.figure(1)
    plt.ylim([1e-10, 1e3])
    plt.xlabel("eigenvalue")
    plt.ylabel("density (log)")
    plt.title(f"full-parameter Hessian ESD evolution (SLQ)\n{cfg.analyze.files_name}")
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spectrum_full_evolution.png"), dpi=150)
    plt.close()

    with open(os.path.join(out_dir, "summary_slq.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for tag in done_tags:
        s = summary[tag]
        print(f"{tag:>5}: lambda_max={s['lambda_max']:.4e}  "
              f"lambda_min={s['lambda_min']:.4e}  tr(H)/D={s['trace_over_dim']:.4e}")
    print("figures written to", out_dir)


def main():
    rank, world, device, is_ddp = setup_dist()
    is_master = rank == 0
    if is_master:
        os.makedirs(out_dir, exist_ok=True)
        for tag, _ in TAGS:
            os.makedirs(os.path.join(out_dir, tag), exist_ok=True)
    if is_ddp:
        dist.barrier()

    batches = load_fixed_batches(device)

    # work items: (tag, probe). Probes are independent -> no communication.
    work = [(tag, k) for (tag, _) in TAGS
            if os.path.exists(os.path.join(run_dir, f"ckpt_{tag}.pt"))
            for k in range(NUM_V)]
    my_work = work[rank::world]

    model_cache = {}

    def get_hessian(tag):
        if tag not in model_cache:
            ckpt = torch.load(os.path.join(run_dir, f"ckpt_{tag}.pt"),
                              map_location=device)
            m = ToyVanilla(model_cfg).to(device).to(DTYPE)
            m.load_state_dict({k: v.to(DTYPE) for k, v in ckpt["model"].items()})
            m.eval()
            model_cache[tag] = FullHessian(m, batches)
        return model_cache[tag]

    for tag, k in my_work:
        path = os.path.join(out_dir, tag, f"ritz_v{k:02d}.npz")
        if os.path.exists(path):
            print(f"[rank {rank}] {tag}/v{k} already done, skipping", flush=True)
            continue
        hess = get_hessian(tag)
        t0 = time.time()
        print(f"[rank {rank}] {tag}/v{k}: Lanczos m={M} on D={hess.dim} ...",
              flush=True)
        evals, weights = lanczos_full_reorth(hess, k, device)
        np.savez(path, values=evals, weights=weights)
        print(f"[rank {rank}] {tag}/v{k} done in {time.time()-t0:.1f}s "
              f"(lambda_max={evals.max():.4e}, lambda_min={evals.min():.4e})",
              flush=True)

    if is_ddp:
        dist.barrier()

    if is_master:
        all_ritz = {}
        for tag, _ in TAGS:
            ritz = []
            for k in range(NUM_V):
                p = os.path.join(out_dir, tag, f"ritz_v{k:02d}.npz")
                if os.path.exists(p):
                    z = np.load(p)
                    ritz.append((z["values"], z["weights"]))
            if ritz:
                all_ritz[tag] = ritz
        if all_ritz:
            render(all_ritz)
        else:
            print("no ritz files produced; check checkpoints in", run_dir)

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
