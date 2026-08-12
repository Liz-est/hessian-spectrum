#!/bin/bash
# 批量提交 3 个 checkpoint 的 DDP 谱计算
cd /data/250010020/hessian-spectrum/QuadraticModel-rep
for pct in 10 50 100; do
    bash submit_spectrum_ddp_p${pct}.sh || echo "p${pct} 提交失败或已存在"
    sleep 2
done
echo "全部提交完成，用 sco acp jobs list 监控"
