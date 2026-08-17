#!/bin/bash
# Muon 端到端 GPU 验证（单节点 8 卡，SCO）。本地 dev 容器 CUDA 驱动过旧无法跑 GPU，
# 故所有 GPU 相关验证走此脚本。依次：
#   1. opt.py 自检（NS5/C₅ 一致性 + CompletePMuon）
#   2. config 切换
#   3. smoke_muon / smoke_adam 各训一个 ckpt（真实 data_grain 管线）
#   4. spectrum_ddp 对 muon ckpt 跑一条 gn_muon（确认 dense-op 预条件器 + eig≥0）
#   5. spectrum_ddp 对 adam ckpt 跑一条 gn_adam（回归，确认没弄坏 Adam 路径）
set -e
cd /data/250010020/hessian-spectrum/QuadraticModel-rep
PY=/data/250010020/miniconda3/envs/nanogpt/bin/python
TR=/data/250010020/miniconda3/envs/nanogpt/bin/torchrun
mkdir -p test_outputs

echo "======== 1. opt.py 自检 ========"; date
$PY opt.py

echo "======== 2. config 切换 ========"
$PY config.py

echo "======== 3a. smoke_muon 训练 ========"; date
$TR --standalone --nproc_per_node=8 train.py smoke_muon
echo "======== 3b. smoke_adam 训练 ========"; date
$TR --standalone --nproc_per_node=8 train.py smoke_adam

echo "======== 4. spectrum_ddp muon ckpt (gn_muon) ========"; date
$TR --standalone --nproc_per_node=8 spectrum_ddp.py \
    --ckpt test_outputs/smoke_muon/ckpt_p100.pt \
    --m 60 --n_tokens 40000 --per 2 --curves gn_muon,gn_raw \
    --out test_outputs/spectrum_smoke_muon.npz

echo "======== 5. spectrum_ddp adam ckpt (gn_adam 回归) ========"; date
$TR --standalone --nproc_per_node=8 spectrum_ddp.py \
    --ckpt test_outputs/smoke_adam/ckpt_p100.pt \
    --m 60 --n_tokens 40000 --per 2 --curves gn_adam,gn_raw \
    --out test_outputs/spectrum_smoke_adam.npz

echo "======== 6. 结果校验 ========"; date
$PY - <<'EOF'
import numpy as np
for tag, f in [("muon","test_outputs/spectrum_smoke_muon.npz"),
               ("adam","test_outputs/spectrum_smoke_adam.npz")]:
    z = np.load(f, allow_pickle=True)
    print(f"[{tag}] optim_name={z['optim_name']}  fields={[k for k in z.files if k.endswith('_eigs')]}")
    for k in z.files:
        if k.endswith("_eigs"):
            e = z[k]
            print(f"    {k}: eig∈[{e.min():.3e},{e.max():.3e}]  min≥0? {e.min()>-1e-6}")
print("✅ 验证脚本完成")
EOF
date
