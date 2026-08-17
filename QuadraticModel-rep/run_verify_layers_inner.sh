#!/bin/bash
# SCO 内部执行：块 Hessian 工具链的验证 + 8 卡真机 smoke test。
# 用法: bash run_verify_layers_inner.sh
# ⚠ 不用 set -e：某一步失败也要继续跑后面的（前两版都因为最后一个小 bug 让
#   步骤 2-5 整个没跑，白等一轮调度）。每步单独记录退出码，末尾汇总。
cd /data/250010020/hessian-spectrum/QuadraticModel-rep

PYTHON=/data/250010020/miniconda3/envs/nanogpt/bin/python
# ⚠ 必须用 conda 里的 torchrun：裸 torchrun 解析到 /usr/local/bin/torchrun（系统
# python3.8），没有 grain/新版 torch → ModuleNotFoundError: No module named 'grain'。
# submit_train_b64.sh 一直用的就是全路径，所以训练没踩到；旧谱脚本写在 grain 依赖
# 引入之前，故也没暴露。
TORCHRUN=/data/250010020/miniconda3/envs/nanogpt/bin/torchrun
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p test_outputs outputs
declare -A RC

# p100 已产出（train-b64-olmo150m SUCCEEDED，val_loss=2.5997）。
CKPT=checkpoints_b64/ckpt_p100.pt

echo "======== 环境 ========"
date; nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true
free -h | head -2

# ---- 1) 数学正确性：小模型与暴力 autograd Hessian/GN 对拍 ----
# 核对 H_bb v == (暴力 H)[blk][:,blk] v、lockstep 逐位一致、闭式 lm_head 块、
# 以及 (H²)_bb ≠ (H_bb)² 的定量证据。单卡即可，GPU 上 1024 次反向很快。
echo ""
echo "======== 1) verify_layers.py（数学对拍）========"
$PYTHON verify_layers.py; RC[1_verify]=$?

# ---- 2) 8 卡 smoke：小 m 跑通分布式 lockstep 全链路 ----
# 只验证「能跑通 + Q 上 GPU + 落盘字段完整」，m 很小，谱本身无意义。
# 跑 gn_raw + gn_adam 两条：两者的预条件器形状不同（raw=0维标量，adam=同形张量），
# 必须都过一遍真机路径。
echo ""
echo "======== 2) 8 卡 lockstep smoke (m=24, raw+adam) ========"
$TORCHRUN --standalone --nproc_per_node=8 -- spectrum_layer_ddp.py \
    --ckpt "$CKPT" \
    --groups embd,head,layer00,layer11 \
    --m 24 --n_tokens 32768 --per 2 \
    --curves gn_raw,gn_adam \
    --out test_outputs/smoke_blocks_p100.npz; RC[2_smoke8]=$?

# ---- 3) full block 交叉验证：块管线的 name=full 应复现 spectrum_ddp 的全量谱 ----
# 两条独立代码路径（hvp.py vs hvp_blocks.py）算同一个算子，谱应一致。
# m=60 足够让 λmax 收敛到几位有效数字，用来对齐量级。
echo ""
echo "======== 3) full block 交叉验证 (m=60) ========"
$TORCHRUN --standalone --nproc_per_node=8 -- spectrum_layer_ddp.py \
    --ckpt "$CKPT" \
    --groups full \
    --m_full 60 --n_tokens 32768 --per 2 \
    --curves gn_raw \
    --out test_outputs/xcheck_full_p100.npz; RC[3_xcheck_block]=$?

echo ""
echo "======== 4) 同参数跑原版全量脚本做对照 ========"
$TORCHRUN --standalone --nproc_per_node=8 -- spectrum_ddp.py \
    --ckpt "$CKPT" \
    --m 60 --n_tokens 32768 --per 2 \
    --curves gn_raw \
    --out test_outputs/xcheck_ref_p100.npz; RC[4_xcheck_ref]=$?

echo ""
echo "======== 5) 对比两条路径的 λmax ========"
$PYTHON - <<'EOF'
import numpy as np
a = np.load("test_outputs/xcheck_full_p100.npz")
b = np.load("test_outputs/xcheck_ref_p100.npz")
ea = a["gn_raw_full_eigs"]; eb = b["gn_raw_eigs"]
print(f"  块管线(full):  λmax={ea.max():.6e}  λmin={ea.min():.3e}  n={ea.size}")
print(f"  原版 spectrum: λmax={eb.max():.6e}  λmin={eb.min():.3e}  n={eb.size}")
rel = abs(ea.max()-eb.max())/max(abs(eb.max()),1e-30)
print(f"  λmax 相对差 {rel:.3e}  → {'✅ 一致' if rel < 5e-2 else '⚠ 需查'}")
# 两条路径的初始向量 seed 不同（块版按 block.name 派生），故只比 λmax 量级，
# 不比逐个 Ritz 值；Krylov 子空间不同，中段 Ritz 值本来就会不同。
EOF
RC[5_compare]=$?

echo ""
echo "======== 6) 逐块作图 smoke（不覆盖既有产出）========"
$PYTHON plot_layers.py --npz test_outputs/smoke_blocks_p100.npz \
    --outdir test_outputs/fig_smoke; RC[6_plot]=$?

echo ""
echo "======== 退出码汇总（0=成功）========"
for k in $(echo "${!RC[@]}" | tr ' ' '\n' | sort); do
  printf "  %-16s %s\n" "$k" "${RC[$k]}"
done
echo "======== 全部完成 ========"; date
