#!/bin/bash
# 提交 Muon lr 扫描（单节点 8 卡，串行 5 个 lr）
cd /data/250010020/hessian-spectrum/QuadraticModel-rep
SCO="/root/.sco/bin/sco --profile zhanglixian-g"
WS="p10-intelligent-adaptation-and-optimization-for-domestic-ai"
out=$($SCO acp jobs create \
  --workspace-name "$WS" \
  --aec2-name "share-cluster" \
  --training-framework "pytorch" \
  --worker-spec "n6ls.iu.i40.8.32c512g" \
  --worker-nodes 1 \
  --container-image-url "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/zkx-ssh-install-g:main-20260515065803" \
  --storage-mount "01995892-d478-76d8-aec7-13fd8284477e:/data" \
  --job-name "sweep-muon-lr" \
  --command "bash /data/250010020/hessian-spectrum/QuadraticModel-rep/run_sweep_muon_lr_inner.sh 2>&1 | tee /data/250010020/hessian-spectrum/QuadraticModel-rep/test_outputs/sweep_muon_lr.log" \
  2>&1)
echo "$out"
