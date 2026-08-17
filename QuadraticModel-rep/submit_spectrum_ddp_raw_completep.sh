#!/bin/bash
# 提交 raw-CompleteP 谱（只 gn_raw/hessian_raw 两条 tag，非真 SGD，用 √(pre·post) 预条件），
# p100，2 节点 × 8 卡。输出独立文件，不覆盖现有结果。
cd /data/250010020/hessian-spectrum/QuadraticModel-rep
SCO="/root/.sco/bin/sco --profile zhanglixian-g"
WS="p10-intelligent-adaptation-and-optimization-for-domestic-ai"

PCT=${1:-p100}
JOB="spectrum-ddp-rawcp-${PCT}"

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
  --command "bash /data/250010020/hessian-spectrum/QuadraticModel-rep/run_spectrum_ddp_raw_completep.sh ${PCT} 2>&1 | tee /data/250010020/hessian-spectrum/QuadraticModel-rep/test_outputs/spectrum_ddp_${PCT}_m1200_raw_completep.log" \
  2>&1)

echo "$out"
