"""
train.py
========
Train the RotatedMatrixBigramProblem replication with a torch optimizer.

This is deterministic full-gradient optimisation of an analytic objective --
no data loading, no DDP; a single GPU (or CPU) per run.  The analytic
`full_grad` from the reference is used as W.grad, so any torch.optim
optimizer drives the exact population dynamics.

Usage
-----
    python train.py <preset-name> [--device=cuda:0]

Outputs under runs/<preset-name>/:
    loss_log.csv           iter, f(W), grad_norm, and per-freq-group losses
    class_losses_<tag>.npy full per-class loss vector at each ckpt fraction
    pi.npy                 the exact Zipf class weights
    config.json            the resolved RunConfig
    loss_curve.png         total loss (log-log)
    group_losses.png       per-frequency-group loss curves
    final_W.pt             only if cfg.save_final_W
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problem as problem_mod
from data import DatasetBigramObjective, load_pairs
from presets import DATASETS, load
from problem import RotatedMatrixBigramProblem

REPL_ROOT = os.path.dirname(os.path.abspath(__file__))


def freq_groups(pi: np.ndarray, n_groups: int):
    """Split classes into n_groups contiguous-by-rank groups of ~equal pi mass.

    pi is already sorted (Zipf by rank), so groups are rank intervals: group 0
    holds the most frequent classes.  Mirrors toy_models/eval_ckpts_val_by_freq.
    """
    cum = np.cumsum(pi)
    edges = [0]
    for g in range(1, n_groups):
        edges.append(int(np.searchsorted(cum, g / n_groups) + 1))
    edges.append(len(pi))
    edges = sorted(set(edges))
    return [np.arange(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    cfg = load(args.preset)
    device = torch.device(args.device)
    dtype = getattr(torch, cfg.dtype)
    torch.manual_seed(0)

    out_dir = os.path.join(REPL_ROOT, "runs", cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=list)

    prob = RotatedMatrixBigramProblem(dim=cfg.dim, embed_init=cfg.embed_init,
                                      embed_std=cfg.embed_std)
    W = prob.get_initialization()
    with torch.no_grad():
        W.data = W.data.to(device, dtype)
    # Materialise E once and pin the cached copy on the target device/dtype so
    # the per-call `.to(W.device, W.dtype)` inside the problem becomes a no-op
    # (otherwise every loss/grad call re-uploads a dim^2 matrix to the GPU).
    print(f"[train] building E (embed_init={cfg.embed_init}, "
          f"std={cfg.embed_std})...", flush=True)
    problem_mod.E_cache[prob.cache_key()] = prob._get_E().to(device, dtype)

    pi = prob._get_probs(W.detach()).cpu().numpy()
    np.save(os.path.join(out_dir, "pi.npy"), pi)
    groups = freq_groups(pi, cfg.n_freq_groups)

    # dataset mode: deterministic full-batch objective over a fixed token set
    dsobj = None
    if cfg.grad_mode == "dataset":
        data_dir = DATASETS[cfg.dataset]
        print(f"[train] loading dataset pairs from {data_dir} ...", flush=True)
        pairs = load_pairs(data_dir)
        assert pairs["V"] == cfg.dim, (pairs["V"], cfg.dim)
        dsobj = DatasetBigramObjective(
            pairs, problem_mod.E_cache[prob.cache_key()], device, dtype)
        rho = dsobj.rho.cpu().numpy()
        floor_np = dsobj.floor.cpu().numpy()
        np.save(os.path.join(out_dir, "rho.npy"), rho)
        np.save(os.path.join(out_dir, "floor.npy"), floor_np)
        print(f"[train] dataset {pairs['name']}: {pairs['n_tokens']} tokens, "
              f"{len(pairs['pair_x'])} unique (x,y) pairs, "
              f"{int((rho > 0).sum())}/{cfg.dim} tokens seen, "
              f"irreducible loss floor = {floor_np.sum():.6f}", flush=True)
    print(f"[train] {cfg.name}: dim={cfg.dim}  opt={cfg.optimizer}  lr={cfg.lr}"
          f"  iters={cfg.max_iters}  grad_mode={cfg.grad_mode}  device={device}")
    if cfg.grad_mode == "stochastic":
        print(f"[train] stochastic: batch_size={cfg.batch_size}  "
              f"data_seed={cfg.data_seed}  (x ~ pi i.i.d., y == x; gradient "
              f"uses the batch's empirical class frequencies)")
    if cfg.embed_init == "orthogonal":
        print(f"[train] Hessian eigs are exactly pi: lambda_max={pi[0]:.4e}  "
              f"lambda_min={pi[-1]:.4e}  cond={pi[0] / pi[-1]:.1f}")
    else:
        # non-orthogonal E: eigs of E^T diag(w) E are not the class weights
        w_h = dsobj.rho if dsobj is not None else prob._get_probs(W.detach())
        lam = prob.hessian_lambda_max(weights=w_h)
        print(f"[train] Hessian = E^T diag(w) E (x) I: lambda_max={lam:.4e}  "
              f"-> GD stable for lr < {2 / lam:.4g}")

    # sampler state for stochastic mode: identity data (y == x) means a batch
    # only enters the loss through its class counts, so sampling a batch ==
    # drawing counts ~ Multinomial(batch_size, pi).
    pi_t = prob._get_probs(W.detach())
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.data_seed)
    pi_cpu = pi_t.cpu()

    # class weights used for grouping/eval: analytic pi (population/stochastic)
    # or the dataset's empirical input-token frequencies rho (dataset mode).
    if dsobj is not None:
        weights = dsobj.rho.cpu().numpy()
        groups = freq_groups(pi, cfg.n_freq_groups)   # rank order == pi order
    else:
        weights = pi

    if cfg.optimizer == "adam":
        opt = torch.optim.Adam([W], lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
                               weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        opt = torch.optim.SGD([W], lr=cfg.lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer}")

    ckpt_iters = {int(round(fr * cfg.max_iters)): fr for fr in cfg.ckpt_fracs}

    log_path = os.path.join(out_dir, "loss_log.csv")
    log_f = open(log_path, "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(["iter", "loss", "grad_norm"]
                    + [f"group{g}_loss" for g in range(len(groups))])

    history = []
    t0 = time.time()
    for it in range(cfg.max_iters + 1):
        with torch.no_grad():
            if cfg.grad_mode == "dataset":
                # count-weighted per-x losses l_x (sum = full-batch objective);
                # per_class = per-token-average loss l_x / rho_x so classes
                # are comparable (0 for tokens absent from the train set)
                lx = dsobj.per_x_losses(W.detach())
                loss = float(lx.sum())
                rho_t = dsobj.rho
                per_class = torch.where(rho_t > 0, lx / rho_t.clamp(min=1e-300),
                                        torch.zeros_like(lx))
                grad = dsobj.grad(W.detach())
            else:
                # evaluation is ALWAYS the population objective (exact pi), so
                # population and stochastic runs are directly comparable
                per_class = prob._per_class_losses(W.detach())
                loss = float((prob._get_probs(W.detach()) * per_class).sum())
                if cfg.grad_mode == "population":
                    grad = prob.full_grad(W.detach())
                else:
                    # fresh batch each step: counts ~ Multinomial(batch_size, pi)
                    counts = torch.multinomial(
                        pi_cpu, cfg.batch_size, replacement=True, generator=gen)
                    freq = torch.bincount(counts, minlength=cfg.dim).to(
                        device, dtype) / cfg.batch_size
                    grad = prob.weighted_grad(W.detach(), freq)
            gnorm = float(grad.norm())

        if it in ckpt_iters:
            tag = f"p{int(round(ckpt_iters[it] * 100)):03d}"
            np.save(os.path.join(out_dir, f"class_losses_{tag}.npy"),
                    per_class.cpu().numpy())

        pc = per_class.cpu().numpy()
        # group loss = weight-averaged within the group (weights: analytic pi,
        # or empirical rho for dataset mode; groups have ~equal pi mass)
        gl = [float((weights[g] * pc[g]).sum() / max(weights[g].sum(), 1e-300))
              for g in groups]
        writer.writerow([it, f"{loss:.10e}", f"{gnorm:.6e}"]
                        + [f"{v:.10e}" for v in gl])
        history.append((it, loss, gl))
        if it % 25 == 0 or it == cfg.max_iters:
            print(f"[train] iter {it:4d}/{cfg.max_iters}  loss {loss:.6e}  "
                  f"|g| {gnorm:.3e}  ({time.time() - t0:.1f}s)", flush=True)

        if it == cfg.max_iters:
            break
        opt.zero_grad(set_to_none=True)
        W.grad = grad
        opt.step()

    log_f.close()
    if cfg.save_final_W:
        torch.save(W.detach().cpu(), os.path.join(out_dir, "final_W.pt"))

    # optimizer description used in every figure title/legend
    if cfg.optimizer == "adam":
        opt_desc = (f"Adam lr={cfg.lr:g} betas={cfg.betas} eps={cfg.eps:g}"
                    f" wd={cfg.weight_decay:g}")
    else:
        opt_desc = (f"SGD lr={cfg.lr:g} mom={cfg.momentum:g}"
                    f" wd={cfg.weight_decay:g}")
    if cfg.grad_mode == "stochastic":
        opt_desc += f" | stoch bs={cfg.batch_size}"
    elif cfg.grad_mode == "dataset":
        opt_desc += f" | dataset={cfg.dataset} full-batch"
    else:
        opt_desc += " | population grad"

    # ---- plots ---------------------------------------------------------- #
    its = np.array([h[0] for h in history])
    losses = np.array([h[1] for h in history])
    plt.figure(figsize=(6, 4))
    plt.plot(its, losses, label=opt_desc)
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("f(W)")
    plt.title(f"{cfg.name}  (final {losses[-1]:.3e})\n{opt_desc}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    gls = np.array([h[2] for h in history])          # (iters, n_groups)
    for g in range(gls.shape[1]):
        plt.plot(its, gls[:, g],
                 label=f"group {g} (ranks {groups[g][0]}-{groups[g][-1]})")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("pi-weighted group loss")
    plt.title(f"{cfg.name}: loss by frequency group (0 = most frequent)\n"
              f"{opt_desc}")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "group_losses.png"), dpi=150)
    plt.close()

    print(f"[done] {cfg.name}: final loss {losses[-1]:.6e}  -> {out_dir}")


if __name__ == "__main__":
    main()
