#!/bin/bash
# SCO worker 内执行：先 tiny 模型 --verify（GPU 上秒级，失败即退），
# 再对给定 ckpt 建 exact 块 Hessian 并落盘 npz。
# 参数: $1=ckpt 路径  $2=输出 npz  [$3=kind, 默认 hessian]
set -e
REP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REP_ROOT"
PY=/data/250010020/miniconda3/envs/nanogpt/bin/python

CKPT=${1:?用法: run_exact_hessian_inner.sh <ckpt> <out.npz> [kind]}
OUT=${2:?缺输出 npz 路径}
KIND=${3:-hessian}

# 批量 VJP（vmap K 条 lane 各持双反向图）显存碎片大，开 expandable_segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== step 1: tiny 模型暴力对拍 ==="
$PY src/analysis/exact_block_hessian.py --verify

NPROC=${NPROC:-8}
echo "=== step 2: exact 块 Hessian  ckpt=$CKPT kind=$KIND nproc=$NPROC ==="
# ⚠ 必须 torchrun：裸 python 在 8 卡机上只用 rank0 那一张（利用率恒 12.5%）
$PY -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    src/analysis/exact_block_hessian.py \
    --ckpt "$CKPT" \
    --layer 1 --param mlp_up --neurons top:8 \
    --kind "$KIND" \
    --n_tokens 65536 --per 8 --chunk 256 \
    --out "$OUT"

echo "=== step 3: 出图 ==="
$PY scripts/plot_exact_hessian.py --npz "$OUT"

echo "✅ done → $OUT"
