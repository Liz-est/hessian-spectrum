"""
frob2 A/B 结果对比图：raw Hessian 谱 keepall(baseline) vs keep99(剔 top-1%) vs 论文缓存。

左：三条谱叠加（combined_positive 逻辑，同 plot_compare_running）。
    - keepall：不过滤（本实验自跑的 baseline，代码路径与 keep99 相同，消除混淆）
    - keep99：剔除 Adam-precond GN-frob2 top-1%
    - Paper B64_P100 hessian_sgd（论文 raw Hessian 缓存）
右：frob2 分数分布（log 直方图）+ q99 阈值线，看重尾有多重、被剔掉多少。

判读：keep99 尾部明显下压向论文 → 二号成立；keepall/keep99 重合 → 二号在真实数据也证伪。

用法：python plot_frob2_compare.py [ab_npz 路径]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.lines import Line2D

N = 167_772_160
AB = sys.argv[1] if len(sys.argv) > 1 else \
    "QuadraticModel-rep/outputs/frob2_ab_p100.npz"
BASELINE = "QuadraticModel-rep/outputs/p100_grain_0815_tensors/full_hessian_raw.npz"
PAPER = "QuadraticModel/analysis/data/cache/spectrum_3x3.npz"

ab = np.load(AB, allow_pickle=True)
base = np.load(BASELINE, allow_pickle=True)  # 不过滤 baseline（已跑过，不重算）
paper = np.load(PAPER, allow_pickle=True)

COL_ALL, COL_99, COL_P = "#888888", "#0072B2", "#d62728"


def get(npz, pre):
    return {k: npz[f"{pre}_{k}"] for k in ["x", "y", "g", "mid", "lo", "hi", "L", "R"]}


def combined_positive(cur):
    """locked(y>0) 与连续体(g>0,mid>0) 合并，按 index 排序去重（同论文）。"""
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


def draw(ax, cur, color, label, band_alpha=0.08):
    x, y = combined_positive(cur)
    ax.plot(x, y, color=color, lw=1.35, solid_capstyle="round", label=label)
    g, mid, lo, hi = cur["g"], cur["mid"], cur["lo"], cur["hi"]
    q = np.isfinite(g) & np.isfinite(mid) & np.isfinite(lo) & np.isfinite(hi) & (g > 0)
    ax.fill_betweenx(g[q], lo[q], hi[q], color=color, alpha=band_alpha, lw=0)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.5))

# ---- 左：三条谱 ----
keepall = get(base, "hessian_raw")   # 不过滤 baseline（已有 npz）
keep99 = get(ab, "keep99")           # 剔 top-1%（本实验）
pc = get(paper, "B64_P100_hessian_sgd")
draw(axL, keepall, COL_ALL, "keepall (no filter)")
draw(axL, keep99, COL_99, "keep99 (drop top-1%)")
draw(axL, pc, COL_P, "Paper B64_P100")
mall = float(combined_positive(keepall)[1].max())
m99 = float(combined_positive(keep99)[1].max())
mp = float(combined_positive(pc)[1].max())
axL.set_title(f"Raw Hessian (CompleteP)\n"
              f"keepall lam_max={mall:.2f}  keep99={m99:.2f}  Paper={mp:.2f}",
              fontsize=10, loc="left")
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlim(0.8, N); axL.set_ylim(1e-8, 5e0)
axL.xaxis.set_major_locator(FixedLocator([1e0, 1e2, 1e4, 1e6, 1e8]))
axL.grid(True, which="both", color="#d4d4d4", alpha=0.55, lw=0.45)
axL.set_xlabel("eigenvalue index (rank)"); axL.set_ylabel("eigenvalue")
axL.legend([Line2D([0], [0], color=c, lw=1.5) for c in (COL_ALL, COL_99, COL_P)],
           ["keepall (no filter)", "keep99 (drop top-1%)", "Paper B64_P100"],
           frameon=False, fontsize=8, loc="lower left")

# ---- 右：frob2 分数分布 ----
scores = np.asarray(ab["frob2_scores"], dtype=np.float64)
thr = float(ab["threshold"])
pos = scores[scores > 0]
n_seqs = int(ab["n_seqs"]); n_kept = len(np.atleast_1d(ab["keep99_idx"]))
bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 60)
axR.hist(pos, bins=bins, color="#0072B2", alpha=0.75)
axR.axvline(thr, color="#d62728", lw=1.5, ls="--", label=f"q99={thr:.2e}")
axR.set_xscale("log"); axR.set_yscale("log")
axR.set_title(f"per-seq Adam-precond GN frob2\n"
              f"n={n_seqs}  kept={n_kept}  removed={n_seqs-n_kept}  "
              f"max/med={scores.max()/np.median(scores):.1f}×", fontsize=10, loc="left")
axR.set_xlabel("frob2 score (‖P G P‖_F² est)"); axR.set_ylabel("count")
axR.grid(True, which="both", color="#d4d4d4", alpha=0.55, lw=0.45)
axR.legend(frameon=False, fontsize=8)

fig.tight_layout()
out = AB.replace(".npz", "_compare.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
print(f"lam_max: keepall={mall:.3f}  keep99={m99:.3f}  paper={mp:.3f}")
print(f"frob2: n={n_seqs} kept={n_kept} removed={n_seqs-n_kept} "
      f"thr={thr:.3e} max={scores.max():.3e} med={np.median(scores):.3e}")
