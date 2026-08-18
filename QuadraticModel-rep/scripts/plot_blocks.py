"""
逐 unit 子块谱作图（读 analyze_blocks.py 的产物 outputs/blocks_<STAMP>/）。

对每个 unit 工作项出：
  esd_<name>.png       该 unit 组的 ESD（pooled 特征值直方图，linear + log 双栏）
  hetero_<name>_skl.png / _js.png   unit×unit 距离热图（unit 数 ≤64 时才画网格，否则热力图）
再出一张跨工作项汇总：
  overview.png         各工作项的 λmax / eff_rank / n_neg / skl_mean 条形对比

逐层的频谱图不与原论文对比（本仓库是自训 checkpoint）。所有图写入 <outdir>，
**已存在则跳过**（除非 --force），不覆盖既有产出。

用法:
  python plot_blocks.py --dir outputs/blocks_p100_0817 [--outdir <默认=<dir>/fig>] [--force]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-12


def _save(fig, path, force):
    if os.path.exists(path) and not force:
        print(f"  [skip] 已存在: {path}（--force 覆盖）")
        plt.close(fig); return
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def fig_esd(name, eigs, info, outpath, force):
    ev = np.asarray(eigs, float).ravel()
    pos = ev[ev > 0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(ev, bins=80, color="steelblue", alpha=0.85)
    ax[0].set_title(f"{name}  ESD (linear)")
    ax[0].set_xlabel("λ"); ax[0].set_ylabel("count")
    if pos.size:
        ax[1].hist(np.log10(pos + EPS), bins=80, color="indianred", alpha=0.85)
    ax[1].set_title(f"{name}  ESD (log10 λ, λ>0)")
    ax[1].set_xlabel("log10 λ")
    s = info.get("pooled", {})
    fig.suptitle(f"{name} | {info.get('unit')} | n_units={info.get('n_units')} "
                 f"d={info.get('d_block')} | λmax={s.get('lam_max',0):.3e} "
                 f"eff_rank={s.get('eff_rank',0):.1f} n_neg={s.get('n_neg',0)}",
                 fontsize=10)
    _save(fig, outpath, force)


def fig_hetero(name, D, metric, outpath, force):
    n = D.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(D, cmap="viridis", aspect="auto")
    ax.set_title(f"{name}  unit×unit {metric.upper()}  (n={n})")
    ax.set_xlabel("unit"); ax.set_ylabel("unit")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, outpath, force)


def fig_overview(summaries, outpath, force):
    names = [s["disp"] for s in summaries]
    lam = [s["pooled"]["lam_max"] for s in summaries]
    eff = [s["pooled"]["eff_rank"] for s in summaries]
    neg = [s["pooled"]["n_neg"] for s in summaries]
    skl = [s.get("skl_mean", 0.0) for s in summaries]
    x = np.arange(len(names))
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for a, vals, t, c in [(ax[0, 0], lam, "λmax", "steelblue"),
                          (ax[0, 1], eff, "eff_rank", "seagreen"),
                          (ax[1, 0], neg, "n_neg (应≈0，GN/Fisher PSD)", "indianred"),
                          (ax[1, 1], skl, "hetero mean (Symmetric-KL)", "darkorange")]:
        a.bar(x, vals, color=c)
        a.set_xticks(x); a.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        a.set_title(t)
        if t == "λmax":
            a.set_yscale("log")
    fig.suptitle("逐 unit 子块谱 概览", fontsize=12)
    fig.tight_layout()
    _save(fig, outpath, force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="analyze_blocks 产出目录")
    ap.add_argument("--outdir", default=None, help="图输出目录（默认 <dir>/fig）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(args.dir, "fig")
    os.makedirs(outdir, exist_ok=True)

    summaries = []
    for sfile in sorted(glob.glob(os.path.join(args.dir, "summary_*.json"))):
        info = json.load(open(sfile))
        disp = info["disp"]
        summaries.append(info)
        eigs = np.load(os.path.join(args.dir, f"eigs_{disp}.npy"))
        print(f"[{disp}] n_units={eigs.shape[0]} d={eigs.shape[1]}")
        fig_esd(disp, eigs, info, os.path.join(outdir, f"esd_{disp}.png"), args.force)
        for metric in ("skl", "js"):
            hf = os.path.join(args.dir, f"hetero_{disp}_{metric}.npy")
            if os.path.exists(hf):
                fig_hetero(disp, np.load(hf), metric,
                           os.path.join(outdir, f"hetero_{disp}_{metric}.png"), args.force)

    if summaries:
        fig_overview(summaries, os.path.join(outdir, "overview.png"), args.force)
    else:
        print("⚠ 目录里没有 summary_*.json")


if __name__ == "__main__":
    main()
