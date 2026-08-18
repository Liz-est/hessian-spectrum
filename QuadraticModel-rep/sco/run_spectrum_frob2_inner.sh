#!/bin/bash
# frob2 A/B 实验内部执行脚本（2 节点 × 8 卡 torchrun），被 submit_spectrum_frob2.sh 调用。
# 验证二号猜疑：Adam-precond GN-frob2 q99 过滤对 raw Hessian 谱的影响。
set -e
REP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REP_ROOT"

PYTHON=/data/250010020/miniconda3/envs/nanogpt/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

CKPT_DIR=checkpoints_b64
OUT_DIR=outputs
mkdir -p "$OUT_DIR" test_outputs

ckpt="${CKPT_DIR}/ckpt_p100.pt"
out="${OUT_DIR}/frob2_ab_p100.npz"

echo "======== frob2 A/B (p100) ========"
date; nvidia-smi --query-gpu=memory.free --format=csv 2>/dev/null || true
free -h | head -2

echo "=== 环境变量检查 ==="
env | grep -E "RANK|WORLD|MASTER|LOCAL|TORCHELASTIC|SENSECORE" || echo "未找到相关环境变量"
echo "===================="

# SCO 多节点：WORLD_SIZE=节点数，RANK=节点编号，每节点 8 卡。
if [ -n "$MASTER_ADDR" ] && [ -n "$RANK" ] && [ -n "$WORLD_SIZE" ]; then
    NNODES=$WORLD_SIZE
    NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-$RANK}
    NPROC=8
    MASTER_PORT=${MASTER_PORT:-29500}
    echo "✅ SCO 多节点: nnodes=$NNODES node_rank=$NODE_RANK nproc=$NPROC master=$MASTER_ADDR:$MASTER_PORT"
    /data/250010020/miniconda3/envs/nanogpt/bin/torchrun \
        --nnodes=$NNODES --nproc_per_node=$NPROC \
        --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        -- src/spectrum/spectrum_frob2_ddp.py --ckpt "$ckpt" \
        --m 1200 --n_tokens 1000000 --per 2 --num_probes 10 \
        --seed 42 --ema 0.04 --out "$out"
else
    echo "⚠️ 单节点回退: 8 卡"
    /data/250010020/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node=8 \
        -- src/spectrum/spectrum_frob2_ddp.py --ckpt "$ckpt" \
        --m 1200 --n_tokens 1000000 --per 2 --num_probes 10 \
        --seed 42 --ema 0.04 --out "$out"
fi

echo "======== frob2 A/B 完成 ========"; date
