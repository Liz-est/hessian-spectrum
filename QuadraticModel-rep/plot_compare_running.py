"""
按论文 combined_positive 逻辑重画：locked 正端 + 连续体合并 → 按 index 排序 → 去重 → 单条线。
解决本仓库 npz 里 locked 与连续体在 index 空间重叠导致的「双线分叉」。
误差带仍来自连续体 lo/hi。叠加论文 B64_P100（同样 combined）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.lines import Line2D

N = 167_772_160
ours = np.load("outputs/spectrum_ddp_p100_m1200_reband.npz", allow_pickle=True)
paper = np.load("../QuadraticModel/analysis/data/cache/spectrum_3x3.npz", allow_pickle=True)
COL_O, COL_P = "#0072B2", "#d62728"


def get(npz, pre):
    return {k: npz[f"{pre}_{k}"] for k in ["x", "y", "g", "mid", "lo", "hi", "L", "R"]}


def combined_positive(cur):
    """论文 combined_positive：locked(y>0) 与连续体(g>0,mid>0) 合并，按 index 排序去重。"""
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


def draw(ax, cur, color, label, band_alpha=0.10):
    x, y = combined_positive(cur)
    ax.plot(x, y, color=color, lw=1.35, solid_capstyle="round", label=label)
    # 误差带来自连续体
    g, mid, lo, hi = cur["g"], cur["mid"], cur["lo"], cur["hi"]
    q = np.isfinite(g) & np.isfinite(mid) & np.isfinite(lo) & np.isfinite(hi) & (g > 0)
    ax.fill_betweenx(g[q], lo[q], hi[q], color=color, alpha=band_alpha, lw=0)


# (curve, title, y-top). raw GN 量级 ~22，需比 precond(~1.8) 高的天花板。
PANELS = [
    ("gn_adam", "Preconditioned Gauss-Newton", 5e0),
    ("hessian_adam", "Preconditioned Hessian", 5e0),
    ("gn_sgd", "Raw Gauss-Newton", 1e2),
    ("hessian_sgd", "Raw Hessian", 1e2),
]
have = [p for p in PANELS if f"{p[0]}_x" in ours.files]
fig, axes = plt.subplots(1, len(have), figsize=(5.2 * len(have), 4.3), squeeze=False)
for ax, (c, t, ytop) in zip(axes[0], have):
    oc = get(ours, c); pc = get(paper, f"B64_P100_{c}")
    draw(ax, oc, COL_O, "Ours")
    draw(ax, pc, COL_P, "Paper B64_P100")
    lo_max = float(combined_positive(oc)[1].max())
    lp_max = float(combined_positive(pc)[1].max())
    ax.set_title(f"{t}\nOurs lam_max={lo_max:.2f}  Paper={lp_max:.2f}", fontsize=10, loc="left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.8, N); ax.set_ylim(1e-8, ytop)
    ax.xaxis.set_major_locator(FixedLocator([1e0, 1e2, 1e4, 1e6, 1e8]))
    ax.grid(True, which="both", color="#d4d4d4", alpha=0.55, lw=0.45)
    ax.legend([Line2D([0], [0], color=COL_O, lw=1.5), Line2D([0], [0], color=COL_P, lw=1.5)],
              ["Ours", "Paper B64_P100"], frameon=False, fontsize=8, loc="lower left")
fig.supxlabel("eigenvalue index (rank)", fontsize=10)
fig.supylabel("eigenvalue", fontsize=10)
fig.tight_layout()
out = "outputs/compare_p100_m1200_v6_reband.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
