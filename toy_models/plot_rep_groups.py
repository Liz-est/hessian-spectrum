"""
plot_rep_groups.py
==================
Per-frequency-group TRAIN loss curves for the REP-* runs under runs/, using
the same plotting logic as replication/ (RotatedMatrixBigramProblem):

  * groups = 10 contiguous-by-rank groups of ~equal pi mass (pi from the
    dataset's meta.pkl, already sorted descending), NOT equal-size groups;
  * per-class loss grouped by the INPUT token x, weighted by the empirical
    input frequency rho_x of the train split;
  * group loss = sum_g rho_x * per_class_x / sum_g rho_x, log-scale y;
  * for the bigram (shift) dataset the per-group irreducible floor
    (optimal row = empirical P(y|x)) is drawn as a dashed line.

All REP runs are 0-layer models (frozen tok_emb + lm_head, optional bias),
so logits = emb @ W.T + b exactly and the full-batch train loss is computed
in closed form from the unique (x, y) pair counts of the train split -- no
model forward, exact to float precision.  The per-position loss follows each
run's own training convention (loss_type "mse" = mean over classes,
"mse_rep" = 0.5 * sum over classes); at W=0 this reproduces the logged
initial train loss (1e-4 resp. 0.5) exactly.

Per run it writes:
    runs/<run>/rep_groups.csv     (tag, iter, total, group_0 .. group_9)
    runs/<run>/rep_groups.png
And per dataset a comparison grid of all runs (converted to the mse_rep
convention so they share one axis):
    runs/rep_groups_compare_<dataset-key>.png
    runs/rep_loss_compare_<dataset-key>.png

Usage:
    python3 plot_rep_groups.py                 # all runs/REP-*
    python3 plot_rep_groups.py adam            # only runs matching substring
    python3 plot_rep_groups.py --mse           # all runs/mse* that have a
                                               # val_by_freq.csv (V1024 bigram);
                                               # their rep_groups.png plots
                                               # EXCESS (loss - floor) curves

For runs trained WITH the sinusoidal pos_enc (older mse0_* runs without the
use_pos_enc=False flag) the closed form uses the expectation over a uniform
block position t (matching the trainer's random-offset sampling):
E_t||A_x+P_t||^2 = ||A_x||^2 + 2 A_x.Pbar + mean_t||P_t||^2 with A = emb@W.T+b
and P = pe@W.T; the cross term uses A + Pbar.  For pos0 runs P == 0 and this
reduces to the exact formula.
"""

import csv
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vanilla_model import sinusoidal_encoding

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RUNS_DIR = os.path.join(HERE, "runs")
N_GROUPS = 10

CKPT_ORDER = ["init", "p10", "p25", "p40", "p50", "p60", "p75", "p85", "p100"]


def freq_groups(pi, n_groups):
    """Contiguous-by-rank groups of ~equal pi mass (replication logic).
    pi must be sorted descending (meta.pkl pi is)."""
    cum = np.cumsum(pi)
    edges = [0]
    for g in range(1, n_groups):
        edges.append(int(np.searchsorted(cum, g / n_groups) + 1))
    edges.append(len(pi))
    edges = sorted(set(edges))
    return [np.arange(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def load_pair_stats(dataset):
    """Unique (x, y) pair counts of the train split + empirical rho + floor.

    For a linear model the full-batch loss depends on the data only through
    these counts, so everything downstream is exact (same trick as
    replication/data.py).  floor is in the mse_rep convention (0.5 * sum).
    """
    data_dir = os.path.join(REPO_ROOT, "data", dataset)
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    pi = np.asarray(meta["pi"], dtype=np.float64)
    V = int(meta["vocab_size"])
    del meta                       # bigram meta holds the dense P (~800 MB)
    x = np.memmap(os.path.join(data_dir, "train_x.bin"), dtype=np.uint16,
                  mode="r").astype(np.int64)
    y = np.memmap(os.path.join(data_dir, "train_y.bin"), dtype=np.uint16,
                  mode="r").astype(np.int64)
    n = len(x)
    uniq, counts = np.unique(x * V + y, return_counts=True)
    px, py, w = uniq // V, uniq % V, counts / n
    rho = np.zeros(V)
    np.add.at(rho, px, w)
    tsq = np.zeros(V)              # ||T_x||^2 = sum_y w_xy^2 per input token
    np.add.at(tsq, px, w * w)
    floor = np.where(rho > 0, 0.5 * (rho - tsq / np.maximum(rho, 1e-300)), 0.0)
    floor[floor < 1e-12] = 0.0     # identity data: exact 0 up to float noise
    return {"V": V, "pi": pi, "n_tokens": n, "rho": rho, "floor": floor,
            "px": torch.from_numpy(px), "py": torch.from_numpy(py),
            "w": torch.from_numpy(w).float()}


def per_x_losses(state, ps, model_cfg=None):
    """Count-weighted per-x train losses, mse_rep convention (0.5 * sum).
    sum() over x == the full-batch train loss.

    If model_cfg says use_pos_enc (older mse0_* runs), the loss is the exact
    EXPECTATION over a uniform block position t of the trainer's random-offset
    batches: with A = emb@W.T (+b) and Q = pos_enc@W.T,
      E_t 0.5*||A_x+Q_t||^2 = 0.5*||A_x+Qbar||^2 + 0.5*(m2 - ||Qbar||^2),
    m2 = mean_t ||Q_t||^2, and the cross term uses A + Qbar."""
    emb = state["tok_emb.weight"].float()
    W = state["lm_head.weight"].float()
    L = emb @ W.T
    bias = state.get("lm_head.bias")
    if bias is not None:
        L += bias.float()
    pos_var = 0.0
    if model_cfg is not None and model_cfg.get("use_pos_enc", True):
        pe = sinusoidal_encoding(model_cfg["block_size"], model_cfg["n_embd"],
                                 torch.device("cpu"))
        Q = pe @ W.T
        qbar = Q.mean(dim=0)
        pos_var = float((Q * Q).sum(dim=1).mean() - qbar @ qbar)
        L += qbar
    rho = torch.from_numpy(ps["rho"]).float()
    row_sq = (L * L).sum(dim=1) + pos_var
    cross = torch.zeros(ps["V"])
    cross.index_add_(0, ps["px"], ps["w"] * L[ps["px"], ps["py"]])
    return (0.5 * rho * row_sq - cross + 0.5 * rho).double().numpy()


def opt_desc(exp):
    o, lr = exp["optim"], exp["lr"]
    if o["name"] in ("adam", "adamw"):
        d = f"{o['name'].capitalize()} lr={lr['learning_rate']:g} " \
            f"betas={tuple(o['betas'])}"
    elif o["name"] == "muon":
        d = f"Muon lr={lr['learning_rate']:g} mom={o['muon_momentum']:g}"
    else:
        d = f"SGD lr={lr['learning_rate']:g} mom={o['momentum']:g}"
    d += f" wd={o['weight_decay']:g}"
    if lr.get("warmup_iters"):
        d += f" | {lr['scheduler']} wu{lr['warmup_iters']}"
    return d


def plot_single_run(run, run_dir, rows, groups, gfloor, desc, dataset,
                    loss_type, excess=False):
    conv = ("mse_rep, 0.5*sum" if loss_type == "mse_rep"
            else "mse, mean over classes")
    its = [r[1] for r in rows]
    plt.figure(figsize=(7.5, 5))
    cmap = plt.cm.viridis
    for g in range(len(groups)):
        color = cmap(g / max(1, len(groups) - 1))
        vals = [r[3][g] for r in rows]
        plt.plot(its, vals, marker="o", ms=3.5, lw=1.6,
                 color=color,
                 label=f"G{g} ranks {groups[g][0]}-{groups[g][-1]}")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel(f"rho-weighted group train loss ({conv})")
    title = f"{run}\n{desc} | ds={dataset}"
    plt.title(title, fontsize=9)
    plt.grid(alpha=0.25, lw=0.5)
    plt.legend(fontsize=7, ncol=2, title="pi-mass groups (0 = most frequent)",
               title_fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(run_dir, "rep_groups.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def process_run(run, cache, force=False, excess=False):
    run_dir = os.path.join(RUNS_DIR, run)
    ckpts = [f"ckpt_{t}.pt" for t in CKPT_ORDER
             if os.path.exists(os.path.join(run_dir, f"ckpt_{t}.pt"))]
    if not ckpts:
        print(f"[{run}] no checkpoints, skipping")
        return None
    ck0 = torch.load(os.path.join(run_dir, ckpts[0]), map_location="cpu",
                     weights_only=False)
    exp = ck0.get("experiment")
    if not exp:
        print(f"[{run}] no experiment config in ckpt, skipping")
        return None
    dataset = exp["data"]["dataset"]
    loss_type = exp["model"]["loss_type"]
    model_cfg = exp["model"]

    # resume: reuse an existing rep_groups.csv (delete it to force recompute)
    csv_path = os.path.join(run_dir, "rep_groups.csv")
    if not force and os.path.exists(csv_path):
        ck0 = None                 # free the ckpt (holds optimizer state, GBs)
        with open(csv_path) as f:
            rd = list(csv.reader(f))
        rows = [(r[0], int(r[1]), float(r[2]), [float(v) for v in r[3:]])
                for r in rd[1:]]
        if len(rows) == len(ckpts):
            print(f"[{run}] reusing {csv_path}")
            if dataset not in cache:
                print(f"[stats] loading pair counts for {dataset} ...",
                      flush=True)
                cache[dataset] = load_pair_stats(dataset)
            ps = cache[dataset]
            groups = freq_groups(ps["pi"], N_GROUPS)
            rho = ps["rho"]
            scale = 2.0 / ps["V"] if loss_type == "mse" else 1.0
            gfloor = [float(ps["floor"][g].sum()
                            / max(rho[g].sum(), 1e-300)) * scale
                      for g in groups]
            plot_single_run(run, os.path.join(RUNS_DIR, run), rows, groups,
                            gfloor, opt_desc(exp), dataset, loss_type,
                            excess=excess)
            return {"run": run, "dataset": dataset, "loss_type": loss_type,
                    "exp": exp, "desc": opt_desc(exp), "rows": rows,
                    "gfloor": gfloor, "V": ps["V"]}

    if dataset not in cache:
        print(f"[stats] loading pair counts for {dataset} ...", flush=True)
        cache[dataset] = load_pair_stats(dataset)
    ps = cache[dataset]
    groups = freq_groups(ps["pi"], N_GROUPS)
    rho = ps["rho"]
    # run's own convention: mse = mean over classes = (2/V) * mse_rep
    scale = 2.0 / ps["V"] if loss_type == "mse" else 1.0

    rows = []
    for name in ckpts:
        t0 = time.time()
        ck = ck0 if name == ckpts[0] else torch.load(
            os.path.join(run_dir, name), map_location="cpu", weights_only=False)
        lx = per_x_losses(ck["model"], ps, model_cfg) * scale
        total = float(lx.sum())
        gl = [float(lx[g].sum() / max(rho[g].sum(), 1e-300)) for g in groups]
        rows.append((ck["tag"], int(ck["iter_num"]), total, gl))
        print(f"[{run}] {ck['tag']:>5s} iter {ck['iter_num']:>4d}  "
              f"total {total:.4e}  G0 {gl[0]:.3e}  G9 {gl[-1]:.3e}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        del ck
    ck0 = None
    rows.sort(key=lambda r: r[1])

    csv_path = os.path.join(run_dir, "rep_groups.csv")
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["tag", "iter", "total_loss"]
                      + [f"group_{g}" for g in range(len(groups))])
        for tag, it, total, gl in rows:
            wcsv.writerow([tag, it, f"{total:.10e}"] + [f"{v:.10e}" for v in gl])

    gfloor = [float(ps["floor"][g].sum() / max(rho[g].sum(), 1e-300)) * scale
              for g in groups]
    desc = opt_desc(exp)
    out_png = plot_single_run(run, run_dir, rows, groups, gfloor, desc,
                              dataset, loss_type, excess=excess)
    print(f"[{run}] wrote {csv_path} and {out_png}")
    return {"run": run, "dataset": dataset, "loss_type": loss_type,
            "exp": exp, "desc": desc, "rows": rows, "gfloor": gfloor,
            "V": ps["V"]}


def compare_plots(results):
    by_ds = {}
    for r in results:
        by_ds.setdefault(r["dataset"], []).append(r)
    for dataset, rs in by_ds.items():
        key = "identity" if "identity" in dataset else "bigram"
        # convert everything to the mse_rep convention for a shared axis
        for r in rs:
            r["k"] = r["V"] / 2.0 if r["loss_type"] == "mse" else 1.0
        # color: Blues = adam, Oranges = sgd, Greens = muon, shade by lr;
        # dashed = no bias
        fams = {"adam": [r for r in rs
                         if r["exp"]["optim"]["name"] in ("adam", "adamw")],
                "sgd": [r for r in rs if r["exp"]["optim"]["name"] == "sgd"],
                "muon": [r for r in rs if r["exp"]["optim"]["name"] == "muon"]}
        cmaps = {"adam": plt.cm.Blues, "sgd": plt.cm.Oranges,
                 "muon": plt.cm.Greens}
        for fam, frs in fams.items():
            lrs = sorted({r["exp"]["lr"]["learning_rate"] for r in frs})
            for r in frs:
                i = lrs.index(r["exp"]["lr"]["learning_rate"])
                r["color"] = cmaps[fam](0.35 + 0.6 * i / max(1, len(lrs) - 1))
                r["ls"] = "--" if "nobias" in r["run"] else "-"

        def label(r):
            lab = r["desc"]
            if "nobias" in r["run"]:
                lab += " | nobias"
            if r["loss_type"] == "mse":
                lab += " | trained on mse"
            return lab

        plt.figure(figsize=(9, 6))
        for r in rs:
            its = [row[1] for row in r["rows"]]
            plt.plot(its, [row[2] * r["k"] for row in r["rows"]],
                     color=r["color"], ls=r["ls"], marker="o", ms=3,
                     label=label(r))
        plt.yscale("log")
        plt.ylim(1e-8, 1e3)   # clamp: diverged runs (1e28/nan) leave the frame
        plt.xlabel("iteration")
        plt.ylabel("train loss (mse_rep convention, 0.5*sum)")
        plt.title(f"REP runs on {dataset}: total train loss\n"
                  f"(mse-trained runs rescaled by V/2 for comparability; "
                  f"dashed = nobias)", fontsize=10)
        plt.grid(alpha=0.25, lw=0.5)
        plt.legend(fontsize=6.5)
        plt.tight_layout()
        out = os.path.join(RUNS_DIR, f"rep_loss_compare_{key}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[plot] {out}")

        n_groups = len(rs[0]["rows"][0][3])
        ncol, nrow = 5, int(np.ceil(n_groups / 5))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow),
                                 sharex=True, sharey=True)
        for g in range(n_groups):
            ax = axes.flat[g]
            for r in rs:
                its = [row[1] for row in r["rows"]]
                ax.plot(its, [row[3][g] * r["k"] for row in r["rows"]],
                        color=r["color"], ls=r["ls"], lw=1.1, label=label(r))
                if r["gfloor"][g] > 0:
                    ax.axhline(r["gfloor"][g] * r["k"], color="gray", ls=":",
                               lw=0.8)
            ax.set_yscale("log")
            ax.set_ylim(1e-8, 1e3)
            ax.set_title(f"freq group {g} (0 = most frequent)", fontsize=9)
            ax.grid(alpha=0.2, lw=0.4)
        for ax in axes.flat[n_groups:]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=5.5)
        fig.suptitle(f"REP runs on {dataset}: rho-weighted group train loss "
                     f"(mse_rep convention; dotted gray = floor)", fontsize=11)
        fig.supxlabel("iteration")
        fig.supylabel("group train loss (0.5*sum)")
        fig.tight_layout()
        out = os.path.join(RUNS_DIR, f"rep_groups_compare_{key}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[plot] {out}")


def sweep_summary():
    """Print excess-loss decay per group for the bigram SGD lr sweep."""
    ps = load_pair_stats('synth_zipf_imbalanced_s1_V10000')
    groups = freq_groups(ps['pi'], N_GROUPS)
    rho = ps['rho']
    scale = 2.0 / ps['V']
    gfloor = np.array([ps['floor'][g].sum() / max(rho[g].sum(), 1e-300)
                       for g in groups]) * scale
    floor_total = float(ps['floor'].sum()) * scale
    print("floor_total =", f"{floor_total:.6e}")
    for t in ["1", "5", "20", "60", "120", "200", "240"]:
        run = f"REP-mse0-pos0-frz_embd-fullbs-sgd-lr{t}-imb-initG02"
        with open(os.path.join(RUNS_DIR, run, "rep_groups.csv")) as f:
            rows = sorted(list(csv.reader(f))[1:], key=lambda r: int(r[1]))
        init = np.array([float(v) for v in rows[0][3:]]) - gfloor
        fin = np.array([float(v) for v in rows[-1][3:]]) - gfloor
        rat = fin / init
        tot = float(rows[-1][2])
        with open(os.path.join(RUNS_DIR, run, "loss_log.csv")) as f:
            ls = sorted([(int(r[0]), float(r[1]))
                         for r in list(csv.reader(f))[1:]])
        vals = [v for _, v in ls]
        nspike = sum(1 for i in range(2, len(vals))
                     if vals[i] > 1.5 * min(vals[:i]) and vals[i] > vals[i - 1])
        mono = all(vals[i + 1] <= vals[i] * 1.02 for i in range(len(vals) - 1))
        print(f"lr={t:>4s} final={tot:.4e} excess={tot - floor_total:.4e} "
              f"nspike={nspike} mono={int(mono)}")
        print("       excess decay G0..G9:",
              " ".join(f"{r:.2e}" for r in rat))


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    if filt == "--sweep-summary":
        sweep_summary()
        return
    if filt == "--mse":
        # all mse* runs that have a val_by_freq figure (V1024 bigram);
        # per-run plots show EXCESS above the irreducible floor
        runs = sorted(d for d in os.listdir(RUNS_DIR)
                      if d.startswith("mse")
                      and os.path.isfile(os.path.join(RUNS_DIR, d,
                                                      "val_by_freq.csv")))
        print(f"[main] {len(runs)} mse* runs")
        cache = {}
        for run in runs:
            process_run(run, cache, excess=True)
        return
    runs = sorted(d for d in os.listdir(RUNS_DIR)
                  if d.startswith("REP") and filt in d
                  and os.path.isdir(os.path.join(RUNS_DIR, d)))
    if not runs:
        sys.exit(f"no REP runs matching '{filt}' under {RUNS_DIR}")
    print(f"[main] {len(runs)} runs")
    cache, results = {}, []
    for run in runs:
        r = process_run(run, cache)
        if r:
            results.append(r)
    if results:
        compare_plots(results)
    # the bigram SGD sweep filter also gets the excess-loss leaderboard
    if "fullbs-sgd-lr" in filt:
        sweep_summary()


if __name__ == "__main__":
    main()
