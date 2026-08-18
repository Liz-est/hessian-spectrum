#!/bin/bash
# 一号猜疑验证：raw Hessian 谱 token 量 1M → 5M（seed=42 同 grain 流，5M⊃1M，可直接对比）。
# 只跑 hessian_raw 单曲线，CompleteP 预条件 √(pre·post)。
# 2 节点 × 8 = 16 卡；5M ≈ 上一轮 1M 的 5×墙钟 ≈ 15-16h。
# 输出独立文件，不覆盖任何现有结果。用法: bash sco/run_spectrum_10m_inner.sh <pct>
set -e
REP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REP_ROOT"

PCT=${1:-p100}
PYTHON=/data/250010020/miniconda3/envs/nanogpt/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT_DIR=checkpoints_b64
OUT_DIR=outputs
mkdir -p "$OUT_DIR" test_outputs

ckpt="${CKPT_DIR}/ckpt_${PCT}.pt"
out="${OUT_DIR}/spectrum_ddp_${PCT}_m1200_hessian_raw_5M.npz"

echo "======== raw Hessian 谱 5M token checkpoint ${PCT}% ========"
date; nvidia-smi --query-gpu=memory.free --format=csv 2>/dev/null || true
free -h | head -2

echo "=== 环境变量检查 ==="
env | grep -E "RANK|WORLD|MASTER|LOCAL|TORCHELASTIC|SENSECORE" || echo "未找到相关环境变量"
echo "===================="

# 只跑 hessian_raw 单曲线；n_tokens=5M；per=2。seed 默认 42（与 1M baseline 同流）。
COMMON="--ckpt $ckpt --m 1200 --n_tokens 5000000 --per 2 \
        --curves hessian_raw --out $out"

if [ -n "$MASTER_ADDR" ] && [ -n "$RANK" ] && [ -n "$WORLD_SIZE" ]; then
    NNODES=$WORLD_SIZE
    NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-$RANK}
    NPROC=8
    MASTER_PORT=${MASTER_PORT:-29500}
    echo "✅ SCO 多节点模式: nnodes=$NNODES node_rank=$NODE_RANK nproc=$NPROC master=$MASTER_ADDR:$MASTER_PORT"
    /data/250010020/miniconda3/envs/nanogpt/bin/torchrun \
        --nnodes=$NNODES --nproc_per_node=$NPROC \
        --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        -- src/spectrum/spectrum_ddp.py $COMMON
else
    echo "⚠️ 单节点回退模式: 8 卡"
    /data/250010020/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node=8 \
        -- src/spectrum/spectrum_ddp.py $COMMON
fi

echo "======== checkpoint ${PCT}% raw Hessian 10M 完成 ========"; date
