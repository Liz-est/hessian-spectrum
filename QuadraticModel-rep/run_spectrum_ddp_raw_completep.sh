#!/bin/bash
# 只重算 raw 两条曲线（tag gn_raw / hessian_raw，非真 SGD），
# 用 CompleteP 预条件 √(pre·post) —— 对齐论文 "raw" 曲线（论文 raw 实为 CompleteP-only，非纯裸 H/G）。
# adam 两条不重算（沿用现有 spectrum_ddp_<pct>_m1200.npz）。
# 输出到独立文件 spectrum_ddp_<pct>_m1200_raw_completep.npz，不覆盖任何现有结果。
# 用法: bash run_spectrum_ddp_raw_completep.sh <pct>
set -e
cd /data/250010020/hessian-spectrum/QuadraticModel-rep

PCT=$1
[ -z "$PCT" ] && { echo "用法: bash run_spectrum_ddp_raw_completep.sh <pct>"; exit 1; }

PYTHON=/data/250010020/miniconda3/envs/nanogpt/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT_DIR=checkpoints_b64
OUT_DIR=outputs
mkdir -p "$OUT_DIR"

ckpt="${CKPT_DIR}/ckpt_${PCT}.pt"
out="${OUT_DIR}/spectrum_ddp_${PCT}_m1200_raw_completep.npz"

echo "======== DDP raw-CompleteP 谱 checkpoint ${PCT}% ========"
date; nvidia-smi --query-gpu=memory.free --format=csv 2>/dev/null || true
free -h | head -2

echo "=== 环境变量检查 ==="
env | grep -E "RANK|WORLD|MASTER|LOCAL|TORCHELASTIC|SENSECORE" || echo "未找到相关环境变量"
echo "===================="

# 只跑 raw 两条曲线（tag gn_raw / hessian_raw）——raw 曲线现固定用 CompleteP 预条件 √(pre·post)。
COMMON="--ckpt $ckpt --m 1200 --n_tokens 1000000 --per 2 \
        --curves gn_raw,hessian_raw --out $out"

if [ -n "$MASTER_ADDR" ] && [ -n "$RANK" ] && [ -n "$WORLD_SIZE" ]; then
    NNODES=$WORLD_SIZE
    NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-$RANK}
    NPROC=8
    MASTER_PORT=${MASTER_PORT:-29500}
    echo "✅ SCO 多节点模式: nnodes=$NNODES node_rank=$NODE_RANK nproc=$NPROC master=$MASTER_ADDR:$MASTER_PORT"
    torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
        --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        -- spectrum_ddp.py $COMMON
else
    echo "⚠️ 单节点回退模式: 8 卡（未检测到多节点环境变量）"
    torchrun --standalone --nproc_per_node=8 -- spectrum_ddp.py $COMMON
fi

echo "======== checkpoint ${PCT}% raw-CompleteP 完成 ========"; date