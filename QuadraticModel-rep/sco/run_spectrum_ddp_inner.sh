#!/bin/bash
# 2 节点 × 8 卡 torchrun 内部执行脚本（被 submit_spectrum_ddp_p100.sh 调用）
# 用法: bash sco/run_spectrum_ddp_inner.sh <pct>
set -e
REP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REP_ROOT"

PCT=$1
[ -z "$PCT" ] && { echo "用法: bash sco/run_spectrum_ddp_inner.sh <pct>"; exit 1; }

PYTHON=/data/250010020/miniconda3/envs/nanogpt/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- A) NCCL 互联诊断：确认跨节点走 IB/RoCE 还是掉到 TCP Socket ----
# NCCL_DEBUG=INFO + SUBSYS=INIT,NET 只打传输层选择（"NET/IB" vs "NET/Socket"），不刷屏。
# 若日志里是 NET/Socket，说明没走高速网 → 640MB×2×1200 步的集合通信是瓶颈根因。
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
echo "=== 网络接口 (ip link) ==="
ip -o link show 2>/dev/null | awk -F': ' '{print $2}' | tr '\n' ' '; echo
echo "=== IB 设备 (ibv_devices) ==="
ibv_devices 2>/dev/null || echo "  无 ibv_devices（可能未装 IB 工具或无 IB）"
echo "=========================="

CKPT_DIR=checkpoints_b64
OUT_DIR=outputs
mkdir -p "$OUT_DIR"

ckpt="${CKPT_DIR}/ckpt_${PCT}.pt"
out="${OUT_DIR}/spectrum_ddp_${PCT}_m1200.npz"

echo "======== DDP 谱计算 checkpoint ${PCT}% ========"
date; nvidia-smi --query-gpu=memory.free --format=csv 2>/dev/null || true
free -h | head -2

# 调试：打印 SCO 设置的所有 PyTorch DDP 相关环境变量
echo "=== 环境变量检查 ==="
env | grep -E "RANK|WORLD|MASTER|LOCAL|TORCHELASTIC|SENSECORE" || echo "未找到相关环境变量"
echo "===================="

# SCO 的变量含义：
# - WORLD_SIZE=2 表示 2 个节点（不是进程数！）
# - RANK=0 表示当前节点编号（node_rank）
# - SENSECORE_PYTORCH_NODE_RANK 也是节点编号
# - 每节点 8 卡，所以总进程数 = WORLD_SIZE × 8
if [ -n "$MASTER_ADDR" ] && [ -n "$RANK" ] && [ -n "$WORLD_SIZE" ]; then
    # SCO 多节点模式：WORLD_SIZE 是节点数，RANK 是节点编号
    NNODES=$WORLD_SIZE
    NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-$RANK}
    NPROC=8
    MASTER_PORT=${MASTER_PORT:-29500}
    echo "✅ SCO 多节点模式: nnodes=$NNODES node_rank=$NODE_RANK nproc=$NPROC master=$MASTER_ADDR:$MASTER_PORT"
    torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
        --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        -- src/spectrum/spectrum_ddp.py --ckpt "$ckpt" --m 1200 --n_tokens 1000000 --per 2 --out "$out"
else
    # 回退单节点 8 卡（sanity）
    echo "⚠️ 单节点回退模式: 8 卡（未检测到多节点环境变量）"
    torchrun --standalone --nproc_per_node=8 \
        -- src/spectrum/spectrum_ddp.py --ckpt "$ckpt" --m 1200 --n_tokens 1000000 --per 2 --out "$out"
fi

echo "======== checkpoint ${PCT}% 完成 ========"; date
