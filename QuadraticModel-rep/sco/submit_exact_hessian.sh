#!/bin/bash
# 提交 exact 块 Hessian job（单节点 8 卡规格，脚本实际只用 1 卡）。
# 用法: bash submit_exact_hessian.sh [CKPT] [KIND] [STAMP]
#   CKPT 默认 checkpoints_b64/ckpt_p50.pt（⚠ 不要用 ckpt_p0：head 零初始化 →
#   mlp_up 等 blocks 的 Hessian 数学上恒为零矩阵）；KIND 默认 hessian；
#   产出 outputs/exact_hessian/<stem>_<KIND>_<STAMP>.npz，不覆盖既有文件。
REP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REP_ROOT"
SCO="/root/.sco/bin/sco --profile zhanglixian-g"
WS="p10-intelligent-adaptation-and-optimization-for-domestic-ai"

CKPT=${1:-checkpoints_b64/ckpt_p50.pt}
KIND=${2:-hessian}
STAMP=${3:-$(date +%m%d_%H%M)}
STEM=$(basename "$CKPT" .pt)
OUT=outputs/exact_hessian/${STEM}_layer01_mlp_up_${KIND}_${STAMP}.npz

if [ ! -f "$CKPT" ]; then
  echo "❌ $CKPT 不存在"
  exit 1
fi
if [ -e "$OUT" ]; then
  echo "❌ $OUT 已存在，换个 STAMP"
  exit 1
fi

JOB="exact-hessian-${STEM//_/-}-${KIND}-${STAMP//_/-}"
LOG="test_outputs/exact_hessian_${STEM}_${KIND}_${STAMP}.log"

out=$($SCO acp jobs create \
  --workspace-name "$WS" \
  --aec2-name "share-cluster" \
  --training-framework "pytorch" \
  --worker-spec "n6ls.iu.i40.8.32c512g" \
  --worker-nodes 1 \
  --container-image-url "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/zkx-ssh-install-g:main-20260515065803" \
  --storage-mount "01995892-d478-76d8-aec7-13fd8284477e:/data" \
  --job-name "$JOB" \
  --command "bash $REP_ROOT/sco/run_exact_hessian_inner.sh $REP_ROOT/$CKPT $REP_ROOT/$OUT $KIND 2>&1 | tee $REP_ROOT/$LOG" \
  2>&1)

echo "$out"
echo "JOB=$JOB  产出→ $OUT  日志→ $LOG"
