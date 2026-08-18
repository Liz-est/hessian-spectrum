#!/bin/bash
# 批量提交 3 个 checkpoint 的 DDP 谱计算
REP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REP_ROOT"
for pct in 10 50 100; do
    bash sco/submit_spectrum_ddp_p${pct}.sh || echo "p${pct} 提交失败或已存在"
    sleep 2
done
echo "全部提交完成，用 sco acp jobs list 监控"
