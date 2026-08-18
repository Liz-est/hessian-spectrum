#!/bin/bash
# B=64 训练：8×H100，olmo150m 到 3B tokens，存 10%/50%/100% checkpoint
REP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
/root/.sco/bin/sco --profile zhanglixian-g acp jobs create \
  --workspace-name "p10-intelligent-adaptation-and-optimization-for-domestic-ai" \
  --aec2-name "share-cluster" \
  --training-framework "pytorch" \
  --worker-spec "n6ls.iu.i40.8.32c512g" \
  --container-image-url "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/zkx-ssh-install-g:main-20260515065803" \
  --storage-mount "01995892-d478-76d8-aec7-13fd8284477e:/data" \
  --job-name "train-b64-olmo150m" \
  --worker-nodes 1 \
  --command "cd $REP_ROOT && mkdir -p test_outputs && /data/250010020/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node=8 src/train/train.py > test_outputs/train_b64.log 2>&1"
