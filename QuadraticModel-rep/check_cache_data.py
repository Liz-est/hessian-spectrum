"""
检查 QuadraticModel 预计算数据的结构
用于理解 Figure 2 的数据格式
"""
import numpy as np
import os

# 预计算数据路径
cache_path = "/data/250010020/hessian-spectrum/QuadraticModel/analysis/data/cache/spectrum_3x3.npz"

if not os.path.exists(cache_path):
    print(f"❌ 文件不存在: {cache_path}")
    exit(1)

print("=" * 80)
print("加载预计算的 Hessian 谱数据...")
print("=" * 80)

# 加载数据
data = np.load(cache_path)

# 列出所有键
print(f"\n数据包含 {len(data.files)} 个键:\n")
for key in sorted(data.files):
    arr = data[key]
    print(f"  {key:50s}  shape={str(arr.shape):20s}  dtype={arr.dtype}")

print("\n" + "=" * 80)
print("Figure 2 需要的数据键（B=64）:")
print("=" * 80)

# Figure 2 显示 4 条曲线：GN, H, GN_P, H_P
# B=64, checkpoint=100% (P100)
target_keys = [
    "B64_P100_gn_identity",      # Raw Gauss-Newton
    "B64_P100_gn_adam",          # Preconditioned Gauss-Newton
    "B64_P100_H_identity",       # Raw Hessian (可能键名不同)
    "B64_P100_H_adam",           # Preconditioned Hessian
]

print("\n查找匹配的键...")
b64_keys = [k for k in data.files if "B64" in k or "b64" in k]
p100_keys = [k for k in data.files if "P100" in k or "p100" in k]

print(f"\n包含 'B64' 的键 ({len(b64_keys)} 个):")
for key in sorted(b64_keys)[:20]:  # 只显示前20个
    print(f"  {key}")

print(f"\n包含 'P100' 的键 ({len(p100_keys)} 个):")
for key in sorted(p100_keys)[:20]:
    print(f"  {key}")

# 查找实际使用的键格式
print("\n" + "=" * 80)
print("推断数据键命名规则:")
print("=" * 80)

# 检查一个示例键的数据结构
sample_keys = [k for k in data.files if "64" in k and ("gn" in k.lower() or "gauss" in k.lower())]
if sample_keys:
    sample_key = sample_keys[0]
    sample_data = data[sample_key]
    print(f"\n示例键: {sample_key}")
    print(f"  Shape: {sample_data.shape}")
    print(f"  Dtype: {sample_data.dtype}")
    print(f"  数据范围: [{sample_data.min():.2e}, {sample_data.max():.2e}]")

    # 如果是一维数组，可能是特征值
    if len(sample_data.shape) == 1:
        print(f"  可能是特征值向量，长度={len(sample_data)}")

    # 显示前10个值
    print(f"  前10个值: {sample_data[:10]}")

print("\n" + "=" * 80)
print("数据键命名模式分析:")
print("=" * 80)

# 分析所有键的命名模式
from collections import defaultdict
patterns = defaultdict(list)

for key in data.files:
    parts = key.split('_')
    # 提取批量大小
    batch_part = [p for p in parts if p.startswith('B') and p[1:].isdigit()]
    # 提取 checkpoint
    ckpt_part = [p for p in parts if p.startswith('P') and p[1:].isdigit()]
    # 提取曲率类型
    curv_type = [p for p in parts if p.lower() in ['gn', 'h', 'gauss', 'hessian']]
    # 提取预条件器
    precond = [p for p in parts if p.lower() in ['adam', 'identity', 'sgd']]

    pattern = f"Batch={batch_part}, Ckpt={ckpt_part}, Curv={curv_type}, Precond={precond}"
    patterns[pattern].append(key)

print("\n发现的命名模式:")
for pattern, keys in sorted(patterns.items()):
    print(f"\n{pattern} ({len(keys)} 个键)")
    for key in keys[:3]:  # 每个模式只显示前3个
        print(f"    {key}")

print("\n" + "=" * 80)
print("✅ 数据结构检查完成")
print("=" * 80)
