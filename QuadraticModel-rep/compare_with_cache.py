"""
对比验证：我们的 B=64 复现 vs 论文缓存 spectrum_3x3.npz

生成并排 Figure 2 对比图（左=缓存，右=复现），计算数值误差指标。

用法：
  python compare_with_cache.py <our_spectrum.npz> --ckpt_frac 1.0 --out figures/
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'font.family': 'serif',
})

CACHE = "/data/250010020/hessian-spectrum/QuadraticModel/analysis/data/cache/spectrum_3x3.npz"
N_PARAMS = 167_772_160
VOCAB = 8192

# 两个子图共享的坐标轴范围（与参考脚本 plot_fig2_from_cache.py 一致）
XLIM = (1, N_PARAMS)
YLIM = (1e-16, 1e0)

SPECTRA = [
    ("gn_adam",      "GN Adam",  "#0072B2", "-"),
    ("hessian_adam", "H Adam",   "#D62728", "-"),
    ("gn_sgd",       "GN raw",   "#0072B2", "--"),
    ("hessian_sgd",  "H raw",    "#D62728", "--"),
]


def yfmt(v, _):
    if v <= 0:
        return "0"
    return rf"$10^{{{int(np.round(np.log10(v)))}}}$"


def draw_spectrum(ax, spec, color, label, linestyle, floor=None):
    """绘制一条谱曲线（与论文 plot_paper_panels.py 一致）。"""
    g, mid, lo, hi = spec["g"], spec["mid"], spec["lo"], spec["hi"]
    L, R = int(spec.get("L", 0)), int(spec.get("R", 0))
    cut = int(spec.get("cut", len(g)))

    # 分段绘制（正/负特征值）
    for where in (slice(None, L) if L > 0 else slice(None, cut),
                  slice(-R, None) if R > 0 else slice(0, 0)):
        q = np.isfinite(g[where]) & np.isfinite(mid[where]) & (g[where] > 0)
        if floor is not None:
            q = q & (g[where] >= floor)
        if q.sum() == 0:
            continue
        gq, mq, lq, hq = g[where][q], mid[where][q], lo[where][q], hi[where][q]
        ax.fill_betweenx(gq, lq, hq, color=color, alpha=0.12, linewidth=0)
        ax.plot(mq, gq, color=color, linestyle=linestyle, linewidth=1.75,
                label=label, solid_capstyle="round")
        label = None  # 只标一次


def load_cache_curve(batch, pct, curvature, precond):
    """从 spectrum_3x3.npz 提取一条曲线。"""
    z = np.load(CACHE)
    prefix = f"B{batch}_P{pct}_{curvature}_{precond}"
    return {k: z[f"{prefix}_{k}"] for k in ("g", "mid", "lo", "hi", "L", "R", "cut")}


def compare_curves(ours, theirs, label):
    """计算两条曲线的数值误差。

    我方 m=400 与缓存 m=1200、mid 轴范围不同，逐索引比会错位。
    正确做法：把 mid 视作 g 的函数 index(g)，在两条曲线**共同的特征值 g 区间**
    上插值到同一 log-g 网格再比。这衡量"给定特征值大小，其 index 位置"，
    与 m 无关，能公平反映头部（大特征值端）是否对齐。
    """
    def clean(spec):
        g, mid = spec["g"], spec["mid"]
        q = np.isfinite(g) & np.isfinite(mid) & (g > 0) & (mid > 0)
        g, mid = g[q], mid[q]
        # g 递减 → index 递增；按 g 升序排一遍便于插值
        order = np.argsort(g)
        return g[order], mid[order]

    g1, mid1 = clean(ours)
    g2, mid2 = clean(theirs)
    if len(g1) < 10 or len(g2) < 10:
        return None

    # 共同 g 区间（头部对齐的部分）
    glo = max(g1.min(), g2.min())
    ghi = min(g1.max(), g2.max())
    if not (ghi > glo):
        return None
    grid = np.geomspace(glo, ghi, 100)
    # index(g)：在 log-log 空间插值（mid 随 g 单调）
    lm1 = np.interp(np.log(grid), np.log(g1), np.log(mid1))
    lm2 = np.interp(np.log(grid), np.log(g2), np.log(mid2))
    m1, m2 = np.exp(lm1), np.exp(lm2)

    rel_err_mid = np.abs(m1 - m2) / (m2 + 1e-16)

    from scipy.stats import spearmanr
    rho_mid, _ = spearmanr(m1, m2)

    return dict(
        label=label, n=len(grid),
        g_overlap_lo=glo, g_overlap_hi=ghi,
        g_range_ours=(g1.min(), g1.max()), g_range_cache=(g2.min(), g2.max()),
        rel_err_mid_med=np.median(rel_err_mid), rel_err_mid_p95=np.percentile(rel_err_mid, 95),
        spearman_mid=rho_mid,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours_npz", help="我们计算的 spectrum npz")
    ap.add_argument("--ckpt_frac", type=float, default=1.0, help="checkpoint 分数（0.1/0.5/1.0）")
    ap.add_argument("--out", default="figures/", help="输出目录")
    args = ap.parse_args()

    batch = 64
    pct = int(args.ckpt_frac * 100)
    ours = np.load(args.ours_npz)

    print(f"对比 B={batch} P={pct} 的 4 条曲线：")
    print(f"  缓存：{CACHE}")
    print(f"  复现：{args.ours_npz}")

    os.makedirs(args.out, exist_ok=True)

    # ---- 并排对比图（左右子图共享同一横纵坐标轴）----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    for i, (src, title) in enumerate([(CACHE, "Paper cache"), (args.ours_npz, "PyTorch replication")]):
        ax = axes[i]
        if i == 0:
            # 缓存：加载 4 条
            curves = [(load_cache_curve(batch, pct, c.split("_")[0], c.split("_")[1]), *s)
                      for c, *s in SPECTRA]
        else:
            # 复现：从 ours 提取
            curves = [({k: ours[f"{c}_{k}"] for k in ("g", "mid", "lo", "hi", "L", "R", "cut")}, *s)
                      for c, *s in SPECTRA]

        for spec, label, color, ls in curves:
            draw_spectrum(ax, spec, color, label, ls)

        ax.axvline(VOCAB, color="0.35", linestyle=":", linewidth=0.85)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ax.yaxis.set_major_formatter(FuncFormatter(yfmt))
        ax.set_xlabel("Eigenvalue Index", fontweight="bold")
        if i == 0:
            ax.set_ylabel("Eigenvalue", fontweight="bold")
        ax.set_title(f"{title}\nB={batch}, {pct}% trained", fontweight="bold")
        ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=9)
        ax.grid(True, which="both", alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    out_png = os.path.join(args.out, f"fig2_comparison_B{batch}_P{pct}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    print(f"\n✓ 对比图：{out_png}")

    # ---- 数值验证 ----
    print("\n数值误差（中位数 / 95分位）：")
    metrics = []
    for curve_name, *_ in SPECTRA:
        ours_curve = {k: ours[f"{curve_name}_{k}"] for k in ("g", "mid", "lo", "hi", "L", "R", "cut")}
        cache_curve = load_cache_curve(batch, pct, curve_name.split("_")[0], curve_name.split("_")[1])
        m = compare_curves(ours_curve, cache_curve, curve_name)
        if m:
            metrics.append(m)
            print(f"  [{m['label']:12s}] 共同g∈[{m['g_overlap_lo']:.2e},{m['g_overlap_hi']:.2e}]  "
                  f"我方g∈[{m['g_range_ours'][0]:.1e},{m['g_range_ours'][1]:.1e}]  "
                  f"缓存g∈[{m['g_range_cache'][0]:.1e},{m['g_range_cache'][1]:.1e}]\n"
                  f"               rel_err(index): {m['rel_err_mid_med']:.1%} / {m['rel_err_mid_p95']:.1%} (中位/p95)  "
                  f"ρ(index)={m['spearman_mid']:.4f}")

    # 保存报告
    report = os.path.join(args.out, f"validation_B{batch}_P{pct}.txt")
    with open(report, "w") as f:
        f.write(f"B={batch} P={pct} 复现验证\n")
        f.write(f"缓存：{CACHE}\n")
        f.write(f"复现：{args.ours_npz}\n\n")
        f.write("对比方法：在两条曲线共同特征值区间上，将 index(g) 插值到同一 log-g "
                "网格后比较（消除 m=400 vs 1200 的采样差异，只比头部对齐度）。\n\n")
        for m in metrics:
            f.write(f"{m['label']:12s}  共同g∈[{m['g_overlap_lo']:.3e},{m['g_overlap_hi']:.3e}]  "
                    f"我方g_max={m['g_range_ours'][1]:.3e} 缓存g_max={m['g_range_cache'][1]:.3e}\n"
                    f"              rel_err(index) med={m['rel_err_mid_med']:.3%} p95={m['rel_err_mid_p95']:.3%}  "
                    f"ρ(index)={m['spearman_mid']:.4f}\n")
    print(f"\n✓ 验证报告：{report}")


if __name__ == "__main__":
    main()
