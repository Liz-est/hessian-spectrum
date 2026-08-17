#!/bin/bash
# 提交块 Hessian 工具链验证（单节点 8×H100）
# 用法: bash submit_verify_layers.sh [job名后缀]
cd /data/250010020/hessian-spectrum/QuadraticModel-rep
SCO="/root/.sco/bin/sco --profile zhanglixian-g"
WS="p10-intelligent-adaptation-and-optimization-for-domestic-ai"

SUFFIX=${1:-}
JOB="verify-blocks${SUFFIX:+-$SUFFIX}"
LOG="test_outputs/verify_layers${SUFFIX:+_$SUFFIX}.log"

if $SCO acp jobs list --workspace-name "$WS" 2>/dev/null | grep -q "$JOB "; then
  echo "job $JOB 已存在，跳过"; exit 0
fi

out=$($SCO acp jobs create \
  --workspace-name "$WS" \
  --aec2-name "share-cluster" \
  --training-framework "pytorch" \
  --worker-spec "n6ls.iu.i40.8.32c512g" \
  --worker-nodes 1 \
  --container-image-url "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/zkx-ssh-install-g:main-20260515065803" \
  --storage-mount "01995892-d478-76d8-aec7-13fd8284477e:/data" \
  --job-name "$JOB" \
  --command "bash /data/250010020/hessian-spectrum/QuadraticModel-rep/run_verify_layers_inner.sh 2>&1 | tee /data/250010020/hessian-spectrum/QuadraticModel-rep/$LOG" \
  2>&1)

echo "$out"
