"""
绘制 Figure 2：B=64 的 Hessian Spectrum
基于预计算数据验证绘图逻辑
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置绘图样式（参考论文）
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

# 加载数据
cache_path = "/data/250010020/hessian-spectrum/QuadraticModel/analysis/data/cache/spectrum_3x3.npz"
data = np.load(cache_path)

# 定义要绘制的曲线（B=64, P100）
# 按论文约定：GN=蓝色，H=红色；Adam=实线，raw(SGD)=虚线
curves = {
    'GN (Adam)': ('B64_P100_gn_adam', '#0072B2', '-'),      # 蓝色实线
    'GN (raw)': ('B64_P100_gn_sgd', '#0072B2', '--'),       # 蓝色虚线
    'H (Adam)': ('B64_P100_hessian_adam', '#d62728', '-'),  # 红色实线
    'H (raw)': ('B64_P100_hessian_sgd', '#d62728', '--'),   # 红色虚线
}

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

print("绘制 Figure 2: B=64 Hessian Spectrum (P100 checkpoint)...")
print("=" * 80)

# 先收集所有数据点来确定横轴范围
all_mid = []

for label, (key_prefix, color, linestyle) in curves.items():
    # 读取网格化数据
    g = data[f'{key_prefix}_g']
    mid = data[f'{key_prefix}_mid']
    lo = data[f'{key_prefix}_lo']
    hi = data[f'{key_prefix}_hi']

    # 过滤有效数据点
    valid = np.isfinite(g) & np.isfinite(mid) & np.isfinite(lo) & np.isfinite(hi) & (g > 0)
    g_plot = g[valid]
    mid_plot = mid[valid]
    lo_plot = lo[valid]
    hi_plot = hi[valid]

    all_mid.extend(mid_plot)

    # 绘制 Gauss-Radau 误差带
    ax.fill_betweenx(g_plot, lo_plot, hi_plot,
                     color=color,
                     alpha=0.10,
                     linewidth=0)

    # 绘制主曲线
    ax.plot(mid_plot, g_plot,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.8)

    print(f"{label:15s}: {len(g_plot):4d} points, "
          f"λ ∈ [{g_plot.min():.2e}, {g_plot.max():.2e}], "
          f"index ∈ [{mid_plot.min():.2e}, {mid_plot.max():.2e}]")

print("=" * 80)

# 设置坐标轴
ax.set_xlabel('Eigenvalue Index', fontweight='bold')
ax.set_ylabel('Eigenvalue', fontweight='bold')
ax.set_title('Figure 2: Hessian Spectrum (B=64, 100% Checkpoint)',
            fontweight='bold', pad=15)

# 设置对数坐标
ax.set_xscale('log')
ax.set_yscale('log')

# 设置坐标范围（从最小 index 开始，即最大特征值的位置）
N_PARAMS = 167_772_160
min_index = min(all_mid) if all_mid else 1
ax.set_xlim(min_index, N_PARAMS)  # 不乘以 0.8
ax.set_ylim(1e-16, 1e0)

# 添加网格
ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)

# 图例
ax.legend(loc='best', framealpha=0.9)

# 保存图像
output_dir = "/data/250010020/hessian-spectrum/QuadraticModel-rep/figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "fig2_b64_p100_from_cache.pdf")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 图像已保存: {output_path}")

# 同时保存 PNG 便于快速查看
output_path_png = output_path.replace('.pdf', '.png')
plt.savefig(output_path_png, dpi=150, bbox_inches='tight')
print(f"✅ PNG 版本: {output_path_png}")

plt.close()

# 额外：对比三个 checkpoint (10%, 50%, 100%)
print("\n" + "=" * 80)
print("绘制三个 checkpoint 的对比图...")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
checkpoints = ['P10', 'P50', 'P100']
checkpoint_labels = ['10%', '50%', '100%']

for ax, ckpt, ckpt_label in zip(axes, checkpoints, checkpoint_labels):
    # 收集当前 checkpoint 的横轴范围
    ckpt_mid = []

    # 绘制 4 条曲线：GN Adam, GN raw, H Adam, H raw
    for curve_label in ['GN (Adam)', 'GN (raw)', 'H (Adam)', 'H (raw)']:
        curve_type = 'gn_adam' if curve_label == 'GN (Adam)' else \
                     'gn_sgd' if curve_label == 'GN (raw)' else \
                     'hessian_adam' if curve_label == 'H (Adam)' else 'hessian_sgd'
        key_prefix = f'B64_{ckpt}_{curve_type}'

        # 读取数据
        g = data[f'{key_prefix}_g']
        mid = data[f'{key_prefix}_mid']
        lo = data[f'{key_prefix}_lo']
        hi = data[f'{key_prefix}_hi']

        # 过滤有效数据
        valid = np.isfinite(g) & np.isfinite(mid) & np.isfinite(lo) & np.isfinite(hi) & (g > 0)
        g_plot = g[valid]
        mid_plot = mid[valid]
        lo_plot = lo[valid]
        hi_plot = hi[valid]

        ckpt_mid.extend(mid_plot)

        # 颜色和线型
        color = '#0072B2' if 'GN' in curve_label else '#d62728'  # 蓝色/红色
        linestyle = '-' if 'Adam' in curve_label else '--'

        # 绘制误差带
        ax.fill_betweenx(g_plot, lo_plot, hi_plot,
                        color=color,
                        alpha=0.08,
                        linewidth=0)

        # 绘制主曲线
        ax.plot(mid_plot, g_plot,
                label=curve_label,
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                alpha=0.8)

    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'{ckpt_label} Training', fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', framealpha=0.9, fontsize=8)
    N_PARAMS = 167_772_160
    min_index_ckpt = min(ckpt_mid) if ckpt_mid else 1
    ax.set_xlim(min_index_ckpt, N_PARAMS)
    ax.set_ylim(1e-16, 1e0)

plt.suptitle('Figure 2: Hessian Spectrum Evolution (B=64, Adam Preconditioned)',
            fontweight='bold', fontsize=14)
plt.tight_layout()

output_path_3ckpt = os.path.join(output_dir, "fig2_b64_3checkpoints_comparison.pdf")
plt.savefig(output_path_3ckpt, dpi=300, bbox_inches='tight')
print(f"\n✅ 三 checkpoint 对比图: {output_path_3ckpt}")

output_path_3ckpt_png = output_path_3ckpt.replace('.pdf', '.png')
plt.savefig(output_path_3ckpt_png, dpi=150, bbox_inches='tight')
print(f"✅ PNG 版本: {output_path_3ckpt_png}")

plt.close()

print("\n" + "=" * 80)
print("✅ 阶段 0 完成：成功读取预计算数据并生成 Figure 2")
print("=" * 80)
print("\n下一步：")
print("1. 检查生成的图像是否与论文 Figure 2 一致")
print("2. 开始实现严格的 Algorithm 1 Lanczos")
print("3. 准备训练数据和模型")
