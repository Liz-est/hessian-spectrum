#!/bin/bash
# 提交 raw Hessian 谱 5M token（一号猜疑验证：token 1M→5M 能否让 rep 谱贴近论文）。
# 2 节点 × 8 = 16 卡，只 hessian_raw 单曲线，CompleteP 预条件。输出独立文件。
REP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REP_ROOT"
SCO="/root/.sco/bin/sco --profile zhanglixian-g"
WS="p10-intelligent-adaptation-and-optimization-for-domestic-ai"

PCT=${1:-p100}
JOB="spectrum-hraw-5m-${PCT}"

if $SCO acp jobs list --workspace-name "$WS" 2>/dev/null | grep -q "$JOB"; then
  echo "job $JOB 已存在，跳过"; exit 0
fi

out=$($SCO acp jobs create \
  --workspace-name "$WS" \
  --aec2-name "share-cluster" \
  --training-framework "pytorch" \
  --worker-spec "n6ls.iu.i40.8.32c512g" \
  --worker-nodes 2 \
  --container-image-url "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/zkx-ssh-install-g:main-20260515065803" \
  --storage-mount "01995892-d478-76d8-aec7-13fd8284477e:/data" \
  --job-name "$JOB" \
  --command "bash $REP_ROOT/sco/run_spectrum_5m_inner.sh ${PCT} 2>&1 | tee $REP_ROOT/test_outputs/spectrum_hraw_5m_${PCT}.log" \
  2>&1)

echo "$out"
