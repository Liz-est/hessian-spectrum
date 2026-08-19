"""
读 exact_block_hessian.py 落盘的 npz，画块 Hessian 结构热图。

计算与画图分离：改样式只重跑本脚本，不重算 H。

五张图：
  1. <stem>_absH.png      |H| 全矩阵 imshow（cividis），vmax 按分位数自适应；
                          叠神经元块分界线
  2. <stem>_blockmean.png 块平均热图：每 (d_in×d_in) 块塌成一个标量 mean|H|，annot 数值
                          —— 只看块间强弱对比，看不到块内结构
  3. <stem>_log10.png     log10|H|，色标锁 [p50, p99.9]；数据跨 6 个量级，这张最可读
  4. <stem>_blocks.png    **逐块全分辨率**：n_sel×n_sel 个子图，每格是一个完整的
                          d_in×d_in Hessian 块本体（各块独立色标，否则强块压死弱块）
  5. <stem>_diagblocks.png 对角块单独一行放大（各神经元自己的块，最常看）

⚠ 色标必须按分位数而非 max：实测 max|H|/median|H| ≈ 1.5e6，`vmax=max/30` 之上
   0.0000% 的元素 → 全部数据挤在色标最底端，整图均匀深蓝、对角非对角看不出差别。
   中位数上 diag/offdiag = 36.6x，对比度是存在的，用 p99.5 才显出来。

用法:
  python scripts/plot_exact_hessian.py --npz outputs/exact_hessian/xxx.npz \
      [--outdir 同npz目录/fig] [--downsample 2048] \
      [--vmax p99.5|1.4e-7|max/30] [--vmin_log p50] [--vmax_log p99.9] \
      [--vmax_block p99.5]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def block_reduce_max(A, ds):
    """|H| 下采样用 max-pool（保留 spike，mean 会把稀疏大元素抹平）。
    ds = 每个输出像素吃进的原始元素数；输出边长 k = ceil(n/ds)。
    ⚠ reshape 必须是 (k, ds, k, ds)：写成 (ds, k, ds, k) 会把 8192 缩成 4×4
    （首版 bug：图只剩角落一个小方块，中间全空）。"""
    n = A.shape[0]
    k = int(np.ceil(n / ds))
    pad = k * ds - n
    if pad:
        A = np.pad(A, ((0, pad), (0, pad)))
    return A.reshape(k, ds, k, ds).max(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--vmax", default="p99.5",
                    help="线性 |H| 色标上限：'p99.5'（分位数）| '1.23e-7'（绝对值）| "
                         "'max/30'（旧行为：max 除以系数）")
    ap.add_argument("--vmin_log", default="p50",
                    help="log10 图色标下限（同 --vmax 格式，默认 p50）")
    ap.add_argument("--vmax_log", default="p99.9",
                    help="log10 图色标上限（同 --vmax 格式，默认 p99.9）")
    ap.add_argument("--vmin_log_block", default="p20",
                    help="逐块 log 图每块独立色标下限（同 --vmax 格式，默认 p20）")
    ap.add_argument("--vmax_log_block", default="p99.5",
                    help="逐块 log 图每块独立色标上限（同 --vmax 格式，默认 p99.5）")
    ap.add_argument("--downsample", type=int, default=2048,
                    help="全矩阵图渲染分辨率上限（max-pool 到 ≤该尺寸）")
    ap.add_argument("--pdf", action="store_true", help="同时存 pdf")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    H = z["H"]
    ids = z["neuron_ids"].tolist()
    d_in = int(z["d_in"])
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    n = H.shape[0]
    n_sel = n // d_in
    stem = os.path.splitext(os.path.basename(args.npz))[0]
    outdir = args.outdir or os.path.join(os.path.dirname(args.npz) or ".", "fig")
    os.makedirs(outdir, exist_ok=True)

    A = np.abs(H)
    amax = A.max()
    title = (f"{meta.get('kind','?')}  layer{meta.get('layer','?')}."
             f"{meta.get('param','?')}  neurons={ids}  "
             f"ckpt={os.path.basename(str(meta.get('ckpt','?')))}")

    def _parse_vmax(spec, arr):
        """解析 --vmax 格式：'p99.5' → 分位数，'1.23e-7' → 绝对值，'max/N' → max/N。"""
        if spec.startswith("p"):
            return float(np.percentile(arr, float(spec[1:])))
        if "/" in spec:
            base, denom = spec.split("/")
            if base == "max":
                return arr.max() / float(denom)
        return float(spec)

    def save(fig, name):
        p = os.path.join(outdir, f"{stem}_{name}.png")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        if args.pdf:
            fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"→ {p}")

    # ---- 1. |H| 全矩阵 ----
    ds = max(1, int(np.ceil(n / args.downsample)))
    Ar = block_reduce_max(A, ds) if ds > 1 else A
    vm = _parse_vmax(args.vmax, A)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(Ar, cmap="cividis", interpolation="nearest",
                   vmin=0, vmax=vm)
    for k in range(1, n_sel):
        ax.axhline(k * d_in / ds - 0.5, color="w", lw=0.5, alpha=0.6)
        ax.axvline(k * d_in / ds - 0.5, color="w", lw=0.5, alpha=0.6)
    ax.set_title(f"|H|  (vmax={args.vmax}={vm:.2e})\n{title}", fontsize=9)
    tick = (np.arange(n_sel) + 0.5) * d_in / ds
    ax.set_xticks(tick); ax.set_xticklabels([f"n{v}" for v in ids], fontsize=7)
    ax.set_yticks(tick); ax.set_yticklabels([f"n{v}" for v in ids], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "absH")

    # ---- 2. 块平均 ----
    B = A.reshape(n_sel, d_in, n_sel, d_in).mean(axis=(1, 3))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(B, cmap="cividis", interpolation="nearest")
    for i in range(n_sel):
        for j in range(n_sel):
            ax.text(j, i, f"{B[i, j]:.1e}", ha="center", va="center",
                    fontsize=6.5, color="w" if B[i, j] < B.max() * 0.6 else "k")
    ax.set_xticks(range(n_sel)); ax.set_xticklabels([f"n{v}" for v in ids], fontsize=8)
    ax.set_yticks(range(n_sel)); ax.set_yticklabels([f"n{v}" for v in ids], fontsize=8)
    diag = np.diag(B).mean()
    offd = (B.sum() - np.trace(B)) / max(1, n_sel * n_sel - n_sel)
    ax.set_title(f"block mean|H|   diag/offdiag = {diag/max(offd,1e-300):.1f}\n{title}",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "blockmean")

    # ---- 3. log10|H| ----
    # 数据跨 6 个数量级（max/p50 ~1.5e6），线性色标怎么调都只能看清一端；
    # log 图把 [p50, p99.9]（~2.5 个量级）铺满色标，块结构最清楚的一张。
    eps = amax * 1e-12
    lo = np.log10(_parse_vmax(args.vmin_log, A) + eps)
    hi = np.log10(_parse_vmax(args.vmax_log, A) + eps)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(np.log10(block_reduce_max(A, ds) + eps) if ds > 1
                   else np.log10(A + eps),
                   cmap="cividis", interpolation="nearest", vmin=lo, vmax=hi)
    for k in range(1, n_sel):
        ax.axhline(k * d_in / ds - 0.5, color="w", lw=0.5, alpha=0.6)
        ax.axvline(k * d_in / ds - 0.5, color="w", lw=0.5, alpha=0.6)
    ax.set_title(f"log10|H|  (range [{args.vmin_log}, {args.vmax_log}] "
                 f"= [{lo:.1f}, {hi:.1f}])\n{title}", fontsize=9)
    ax.set_xticks(tick); ax.set_xticklabels([f"n{v}" for v in ids], fontsize=7)
    ax.set_yticks(tick); ax.set_yticklabels([f"n{v}" for v in ids], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "log10")

    print(f"n={n}  d_in={d_in}  max|H|={amax:.3e}  "
          f"diag/offdiag(blockmean)={diag/max(offd,1e-300):.1f}  "
          f"sym_resid={meta.get('sym_resid')}")

    # ---- 4. 逐块细节：每个 d_in×d_in 块单独一格，全分辨率（不下采样）----
    # 块平均只给 8×8 个标量，看不到块**内部**长什么样；这张图才是"每个 block
    # 的 Hessian 本体"。各块独立色标（按块内分位数，不用块内 max：单个 spike
    # 元素能比中位数大 6 个量级，用 max 会让整块变成均匀深色）。
    # 块内也必须走 log：块内 bulk 很平（p50→p90 仅 2.5x）但 max/p50 达 3e2~3e4，
    # 且约一半的大元素挤在 1024 行里的 10 行 → 线性标只能显示"平底 + 几根亮线"。
    def _blk_log(blk):
        lo = np.log10(_parse_vmax(args.vmin_log_block, blk) + eps)
        hi = np.log10(_parse_vmax(args.vmax_log_block, blk) + eps)
        return np.log10(blk + eps), lo, (hi if hi > lo else lo + 1.0)

    fig, axes = plt.subplots(n_sel, n_sel, figsize=(1.6 * n_sel, 1.6 * n_sel))
    for i in range(n_sel):
        for j in range(n_sel):
            blk = A[i * d_in:(i + 1) * d_in, j * d_in:(j + 1) * d_in]
            ax = axes[i, j]
            L, lo, hi = _blk_log(blk)
            ax.imshow(L, cmap="cividis", interpolation="nearest", vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"n{ids[j]}", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"n{ids[i]}", fontsize=7)
    fig.suptitle(f"per-block log10|H| full-res ({d_in}x{d_in}/block, per-block scale "
                 f"[{args.vmin_log_block}, {args.vmax_log_block}])\n{title}", fontsize=9)
    save(fig, "blocks")

    # ---- 5. 对角块单独放大（最常看的 8 张）----
    fig, axes = plt.subplots(1, n_sel, figsize=(2.6 * n_sel, 3.0))
    for k, ax in enumerate(np.atleast_1d(axes)):
        blk = A[k * d_in:(k + 1) * d_in, k * d_in:(k + 1) * d_in]
        L, lo, hi = _blk_log(blk)
        im = ax.imshow(L, cmap="cividis", interpolation="nearest", vmin=lo, vmax=hi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"n{ids[k]}\nmax={blk.max():.1e}  p50={np.median(blk):.1e}", fontsize=7)
    fig.suptitle(f"diagonal blocks, log10|H| (each neuron's own {d_in}x{d_in} Hessian, "
                 f"per-block scale [{args.vmin_log_block}, {args.vmax_log_block}])\n{title}",
                 fontsize=9)
    save(fig, "diagblocks")


if __name__ == "__main__":
    main()
