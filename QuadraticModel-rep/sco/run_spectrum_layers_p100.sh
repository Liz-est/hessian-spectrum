#!/bin/bash
# 生产跑：p100 的全量 + 逐层 Hessian/GN 谱（2 节点 × 8 卡 = 16 卡）。
#
# 曲线顺序按要求：raw Hessian → preconditioned Hessian → raw GN → preconditioned GN
# 每条曲线跑完**立即作图**（全量图 + 逐块图），再进入下一条。
#
# ⚠ 不覆盖既有产出：所有输出进 outputs/p100_grain_<STAMP>/，STAMP 由提交时传入；
#   作图脚本本身也有「已存在则跳过」保护（需 --force 才覆盖）。
#
# 用法: bash sco/run_spectrum_layers_p100.sh <STAMP> [CKPT]
REP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REP_ROOT"

PYTHON=/data/250010020/miniconda3/envs/nanogpt/bin/python
# ⚠ 必须用 conda 里的 torchrun（裸 torchrun → 系统 python3.8，没有 grain）
TORCHRUN=/data/250010020/miniconda3/envs/nanogpt/bin/torchrun
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP=${1:?用法: run_spectrum_layers_p100.sh <STAMP> [CKPT]}
CKPT=${2:-checkpoints_b64/ckpt_p100.pt}
OUT=outputs/p100_grain_${STAMP}
FIG=${OUT}/fig
mkdir -p "$OUT" "$FIG"

if [ ! -f "$CKPT" ]; then
  echo "❌ 找不到 checkpoint: $CKPT"; exit 1
fi

# 全量谱的 m：与既有 p100 run 对齐（m=1200）
M_FULL=1200
# 逐块谱的 m：块只有 4.2~8.4M 维（全量 167.8M 的 1/40~1/20），极端 Ritz 值收敛快得多。
# 收敛判据见 gauss_radau.ritz_convergence 的 rel = β_m|U[-1,i]|/|θ_i|，跑完可查 L。
M_BLOCK=400
# 块粒度：layer-tensors = embd + head + 每层 6 个张量各一块（74 块）：
#   attn_q/attn_k/attn_v (=wq/wk/wv)、attn_head (=attn_proj/o_proj)、
#   mlp_up (=mlp_fc)、mlp_head (=mlp_proj)
# 参考 vision_models/hessian_spectrum.py 的 get_spectrum_layer_by_layer：
# 那边模型每层是独立 nn.Linear，named_parameters() 天然给到张量粒度；本仓库参数
# 跨层 stack（attn_q 是单个 (L,D,H,K) 叶子），所以要显式按 L 轴切片。
# ⚠ 变量名不能叫 GROUPS：bash 里 GROUPS 是内建特殊只读数组（进程附属组 ID），
# 给它赋标量会走算术上下文，`layer-tensors` 被当 `layer - tensors = 0 - 0 = 0`，
# 于是 --layers 收到 "0" → build_layers 报「未知 block token: 0」→ 逐块步全崩。
# （0815 首个生产 job 的逐块谱就是这么全灭的，全量谱不受影响。）
LAYER_GROUPS=layer-tensors
# Q 显存：Σn_b 恒等于全模型，故与粒度无关 —— m=400/world=16 时 16.8GB/rank，
# 加 HVP 峰值保留 10.8GB = 27.6/80GB，可整批 lockstep 不分波。
MAX_LOCKSTEP=0
N_TOKENS=1000000
PER=2

# torchrun 拼装（SCO 多节点：WORLD_SIZE=节点数，RANK=节点编号）
if [ -n "$MASTER_ADDR" ] && [ -n "$RANK" ] && [ -n "$WORLD_SIZE" ]; then
    TR="$TORCHRUN --nnodes=$WORLD_SIZE --nproc_per_node=8 \
        --node_rank=${SENSECORE_PYTORCH_NODE_RANK:-$RANK} \
        --master_addr=$MASTER_ADDR --master_port=${MASTER_PORT:-29500}"
    echo "✅ 多节点: nnodes=$WORLD_SIZE node_rank=${SENSECORE_PYTORCH_NODE_RANK:-$RANK}"
else
    TR="$TORCHRUN --standalone --nproc_per_node=8"
    echo "⚠️ 单节点回退（8 卡）"
fi

echo "======== 配置 ========"
echo "  ckpt=$CKPT  out=$OUT"
echo "  m_full=$M_FULL  m_block=$M_BLOCK  n_tokens=$N_TOKENS  per=$PER"
date; nvidia-smi --query-gpu=memory.free --format=csv 2>/dev/null | head -3
free -h | head -2

declare -A RC

# 曲线顺序：raw Hessian → precond Hessian → raw GN → precond GN
for TAG in hessian_raw hessian_adam gn_raw gn_adam; do
  echo ""
  echo "############################################################"
  echo "# 曲线 $TAG  开始   $(date)"
  echo "############################################################"

  # ---- (a) 全量谱 ----
  FULL_NPZ="${OUT}/full_${TAG}.npz"
  if [ -f "$FULL_NPZ" ]; then
    echo "  [skip] 已存在 $FULL_NPZ"
  else
    $TR -- src/spectrum/spectrum_ddp.py \
        --ckpt "$CKPT" --m $M_FULL --n_tokens $N_TOKENS --per $PER \
        --curves "$TAG" --out "$FULL_NPZ"
    RC[${TAG}_full]=$?
  fi

  # ---- (b) 逐层/张量谱（embd + head + 每层 attn/mlp 各一块）----
  BLK_NPZ="${OUT}/layers_${TAG}.npz"
  if [ -f "$BLK_NPZ" ]; then
    echo "  [skip] 已存在 $BLK_NPZ"
  else
    $TR -- src/spectrum/spectrum_layer_ddp.py \
        --ckpt "$CKPT" --layers "$LAYER_GROUPS" \
        --m $M_BLOCK --n_tokens $N_TOKENS --per $PER \
        --max_lockstep $MAX_LOCKSTEP \
        --curves "$TAG" --out "$BLK_NPZ"
    RC[${TAG}_layers]=$?
  fi

  # ---- (c) 立即作图（只 rank0 需要，torchrun 外单进程跑）----
  echo "  --- 作图 $TAG ---"
  # 全量：不与论文对比（新 run 是 100BT parquet 口径，与论文缓存不同源）
  $PYTHON scripts/plot_full_spectrum.py --npz "$FULL_NPZ" \
      --outdir "$FIG" --name "full_${TAG}"
  RC[${TAG}_plot_full]=$?
  # 逐块：overlay + grid + summary
  $PYTHON scripts/plot_layers.py --npz "$BLK_NPZ" --outdir "$FIG"
  RC[${TAG}_plot_layers]=$?

  echo "# 曲线 $TAG  完成   $(date)"
done

# ---- 四条曲线齐了，再出一张 2×2 汇总图 ----
echo ""
echo "======== 四曲线汇总图 ========"
$PYTHON scripts/plot_full_spectrum.py --npz ${OUT}/full_*.npz \
    --outdir "$FIG" --name "full_all4"
RC[all4]=$?

echo ""
echo "======== 退出码汇总（0=成功）========"
for k in $(echo "${!RC[@]}" | tr ' ' '\n' | sort); do
  printf "  %-22s %s\n" "$k" "${RC[$k]}"
done
echo "产出目录: $OUT"
ls -la "$OUT" "$FIG" 2>/dev/null
echo "======== 全部完成 ========"; date
