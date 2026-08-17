"""
全量谱作图（参数化路径版），逻辑与 plot_compare_running.py 一致但不写死 npz 路径，
且**不修改**既有脚本、不覆盖既有产出。

与论文的对比是可选的（--paper）：新 run（100BT parquet 数据）与论文缓存的数据口径已不同
（非 bit 级复现，见 data_grain.py），默认不叠论文曲线；需要时显式打开。

支持"每跑完一条曲线就画"：只画 npz 里已存在的 tag，缺的跳过。

用法:
  python plot_full_spectrum.py --npz outputs/p100_grain/full_hessian_raw.npz \
      --outdir outputs/p100_grain/fig
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

COL_O, COL_P = "#0072B2", "#d62728"

# (our-tag, title, paper-key-suffix)
# ⚠ 论文缓存把 raw 两条写死叫 gn_sgd/hessian_sgd（实为 CompleteP 预条件，非真 SGD），
#   key 不可改，只在此处映射。见 memory: quadraticmodel-sgd-means-raw-completep
PANELS = [
    ("hessian_raw", "Raw Hessian (CompleteP)", "hessian_sgd"),
    ("hessian_adam", "Preconditioned Hessian", "hessian_adam"),
    ("gn_raw", "Raw Gauss-Newton (CompleteP)", "gn_sgd"),
    ("gn_adam", "Preconditioned Gauss-Newton", "gn_adam"),
]


def get(npz, pre):
    return {k: npz[f"{pre}_{k}"] for k in ["x", "y", "g", "mid", "lo", "hi", "L", "R"]}


def combined_positive(cur):
    """与 plot_compare_running.combined_positive 同逻辑。"""
    L = int(np.atleast_1d(cur["L"])[0])
    xs, ys = [], []
    if L:
        xl, yl = cur["x"][:L], cur["y"][:L]
        q = np.isfinite(xl) & np.isfinite(yl) & (yl > 0)
        xs.append(xl[q]); ys.append(yl[q])
    g, mid = cur["g"], cur["mid"]
    q = np.isfinite(g) & np.isfinite(mid) & (g > 0) & (mid > 0)
    xs.append(mid[q]); ys.append(g[q])
    x = np.concatenate(xs); y = np.concatenate(ys)
    order = np.argsort(x); x, y = x[order], y[order]
    uniq = np.r_[True, np.diff(x) > 0]
    return x[uniq], y[uniq]


def draw(ax, cur, color, band_alpha=0.10):
    x, y = combined_positive(cur)
    ax.plot(x, y, color=color, lw=1.35, solid_capstyle="round")
    g, mid, lo, hi = cur["g"], cur["mid"], cur["lo"], cur["hi"]
    q = np.isfinite(g) & np.isfinite(mid) & np.isfinite(lo) & np.isfinite(hi) & (g > 0)
    ax.fill_betweenx(g[q], lo[q], hi[q], color=color, alpha=band_alpha, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, nargs="+",
                    help="一个或多个 npz（多个时按给出顺序合并，后者覆盖同名 tag）")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--name", default="full_spectrum")
    ap.add_argument("--paper", default="",
                    help="论文缓存 spectrum_3x3.npz 路径；给了才叠论文曲线")
    ap.add_argument("--ytop", type=float, default=5e0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    merged, n_params = {}, None
    for p in args.npz:
        z = np.load(p, allow_pickle=True)
        for f in z.files:
            merged[f] = z[f]
        if "n_params" in z.files:
            n_params = int(z["n_params"])
    N = n_params or 167_772_160

    class _M:
        files = list(merged.keys())
        def __getitem__(self, k):
            return merged[k]
    src = _M()

    paper = np.load(args.paper, allow_pickle=True) if args.paper else None
    have = [p for p in PANELS if f"{p[0]}_x" in src.files]
    if not have:
        raise SystemExit(f"npz 里没有任何完整曲线：{args.npz}")
    print(f"可画曲线: {[h[0] for h in have]}  (N={N:,})", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.name}.png")
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"已存在（加 --force 才覆盖）: {out}")

    fig, axes = plt.subplots(1, len(have), figsize=(5.2 * len(have), 4.3), squeeze=False)
    for ax, (tag, title, pkey) in zip(axes[0], have):
        oc = get(src, tag)
        draw(ax, oc, COL_O)
        lo_max = float(combined_positive(oc)[1].max())
        sub = f"Ours lam_max={lo_max:.3g}"
        labels, handles = ["Ours"], [Line2D([0], [0], color=COL_O, lw=1.5)]
        if paper is not None and f"B64_P100_{pkey}_x" in paper.files:
            pc = get(paper, f"B64_P100_{pkey}")
            draw(ax, pc, COL_P)
            sub += f"  Paper={float(combined_positive(pc)[1].max()):.3g}"
            labels.append("Paper B64_P100")
            handles.append(Line2D([0], [0], color=COL_P, lw=1.5))
        ax.set_title(f"{title}\n{sub}", fontsize=10, loc="left")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.8, N); ax.set_ylim(1e-8, args.ytop)
        ax.xaxis.set_major_locator(FixedLocator([1e0, 1e2, 1e4, 1e6, 1e8]))
        ax.grid(True, which="both", color="#d4d4d4", alpha=0.55, lw=0.45)
        ax.legend(handles, labels, frameon=False, fontsize=8, loc="lower left")
    fig.supxlabel("eigenvalue index (rank)", fontsize=10)
    fig.supylabel("eigenvalue", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out, flush=True)


if __name__ == "__main__":
    main()
