"""
按论文 combined_positive 逻辑重画：locked 正端 + 连续体合并 → 按 index 排序 → 去重 → 单条线。
误差带来自连续体 lo/hi。叠加论文 B64_P100（同样 combined）。

raw 两条曲线（我们内部 tag gn_raw/hessian_raw，非真 SGD）：论文的 "raw" 实为 CompleteP 预条件
√(pre·post)，非纯裸 H/G（已验证：CompleteP 后 λmax≈1.2，与论文 raw≈0.34 同量级 ~4×；纯裸会到
~22, 64×）。故 raw 面板从 CompleteP 版 npz 取数（spectrum_ddp_p100_m1200_raw_completep.npz）。
⚠ 论文缓存里这两条写死叫 B64_P100_gn_sgd/hessian_sgd（别人的数据、key 不可改），故仅在本脚本里
把我们的 raw tag 映射到论文的 sgd key（见 PANELS 的 paper_key 字段）。adam 两条不变。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.lines import Line2D
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import paths


N = 167_772_160
ours = np.load(paths.OUT_DIR / "spectrum_ddp_p100_m1200_hessian_raw_5M.npz", allow_pickle=True)
paper = np.load(paths.CACHE_NPZ, allow_pickle=True)
# raw 曲线固定从 CompleteP 版 npz 取（我们内部 tag gn_raw/hessian_raw）
# _CP_PATH = "111"
# if not os.path.exists(_CP_PATH):
#     raise SystemExit(f"缺少 CompleteP raw 谱 {_CP_PATH}（先跑 run_spectrum_ddp_raw_completep.sh）")
# ours_cp = np.load(_CP_PATH, allow_pickle=True)
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


# (our-tag, title, y-top, source-npz, paper-key-suffix). raw 用 CompleteP 版 → λ 被压到 O(1)，y-top 5e0。
# paper-key-suffix：论文缓存写死的 key 后缀（B64_P100_<suffix>）；raw 曲线论文那边叫 sgd。
PANELS = [
    ("gn_adam", "Preconditioned Gauss-Newton", 5e0, ours, "gn_adam"),
    ("hessian_adam", "Preconditioned Hessian", 5e0, ours, "hessian_adam"),
    ("gn_raw", "Raw Gauss-Newton (CompleteP)", 5e0, ours, "gn_sgd"),
    ("hessian_raw", "Raw Hessian (CompleteP)", 5e0, ours, "hessian_sgd"),
]

have = [p for p in PANELS if f"{p[0]}_x" in p[3].files]
fig, axes = plt.subplots(1, len(have), figsize=(5.2 * len(have), 4.3), squeeze=False)
for ax, (c, t, ytop, src, pkey) in zip(axes[0], have):
    oc = get(src, c); pc = get(paper, f"B64_P100_{pkey}")
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
out = str(paths.ensure(paths.OUT_DIR) / "compare_p100_m1200_5M.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
