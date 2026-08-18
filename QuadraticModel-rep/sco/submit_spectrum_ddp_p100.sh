#!/bin/bash
# 提交 p100 谱计算（2 节点 × 8 卡，1M token + m=800）
REP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REP_ROOT"
SCO="/root/.sco/bin/sco --profile zhanglixian-g"
WS="p10-intelligent-adaptation-and-optimization-for-domestic-ai"

# 检查是否已提交
if $SCO acp jobs list --workspace-name "$WS" 2>/dev/null | grep -q "spectrum-ddp-p100"; then
  echo "job 已存在，跳过"; exit 0
fi

out=$($SCO acp jobs create \
  --workspace-name "$WS" \
  --aec2-name "share-cluster" \
  --training-framework "pytorch" \
  --worker-spec "n6ls.iu.i40.8.32c512g" \
  --worker-nodes 2 \
  --container-image-url "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/zkx-ssh-install-g:main-20260515065803" \
  --storage-mount "01995892-d478-76d8-aec7-13fd8284477e:/data" \
  --job-name "spectrum-ddp-p100" \
  --command "bash $REP_ROOT/sco/run_spectrum_ddp_inner.sh p100 2>&1 | tee $REP_ROOT/test_outputs/spectrum_ddp_p100_m1200.log" \
  2>&1)

echo "$out"
