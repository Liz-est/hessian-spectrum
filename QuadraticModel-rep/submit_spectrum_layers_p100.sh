#!/bin/bash
# 提交 p100 全量+逐层谱分析（2 节点 × 8 卡 = 16 卡）。
#
# ⚠ 前置：train-b64-olmo150m 必须已 SUCCEEDED 且 checkpoints_b64/ckpt_p100.pt 存在。
#   本脚本会检查，不满足直接退出（不提交空跑的 job）。
#
# 用法: bash submit_spectrum_layers_p100.sh [STAMP]
#   STAMP 默认取当前时间戳；产出进 outputs/p100_grain_<STAMP>/，绝不覆盖既有目录。
cd /data/250010020/hessian-spectrum/QuadraticModel-rep
SCO="/root/.sco/bin/sco --profile zhanglixian-g"
WS="p10-intelligent-adaptation-and-optimization-for-domestic-ai"

STAMP=${1:-$(date +%m%d_%H%M)}
CKPT=checkpoints_b64/ckpt_p100.pt
OUT=outputs/p100_grain_${STAMP}

if [ ! -f "$CKPT" ]; then
  echo "❌ $CKPT 不存在 —— train-b64-olmo150m 还没跑完，先等它 SUCCEEDED"
  exit 1
fi
if [ -e "$OUT" ]; then
  echo "❌ $OUT 已存在，换个 STAMP（不覆盖既有产出）"
  exit 1
fi

JOB="spectrum-blocks-p100-${STAMP//_/-}"
LOG="test_outputs/spectrum_layers_p100_${STAMP}.log"

out=$($SCO acp jobs create \
  --workspace-name "$WS" \
  --aec2-name "share-cluster" \
  --training-framework "pytorch" \
  --worker-spec "n6ls.iu.i40.8.32c512g" \
  --worker-nodes 2 \
  --container-image-url "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/zkx-ssh-install-g:main-20260515065803" \
  --storage-mount "01995892-d478-76d8-aec7-13fd8284477e:/data" \
  --job-name "$JOB" \
  --command "bash /data/250010020/hessian-spectrum/QuadraticModel-rep/run_spectrum_layers_p100.sh $STAMP 2>&1 | tee /data/250010020/hessian-spectrum/QuadraticModel-rep/$LOG" \
  2>&1)

echo "$out"
echo "STAMP=$STAMP  产出→ $OUT  日志→ $LOG"
