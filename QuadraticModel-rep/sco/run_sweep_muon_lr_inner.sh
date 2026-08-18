#!/bin/bash
# Muon lr 扫描（单节点 8 卡）：5 个 lr × 3200 步，同 full 45776 步 cosine schedule（公平对比 Adam 早段）。
# Adam(opt_lr=16) 基准：step1526→3.58, step3052→3.36。串行跑，末尾汇总对比。
set -e
REP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REP_ROOT"
TR=/data/250010020/miniconda3/envs/nanogpt/bin/torchrun
PY=/data/250010020/miniconda3/envs/nanogpt/bin/python
mkdir -p test_outputs

for LR in 80 160 240 320; do
  echo "======== Muon lr=$LR ========"; date
  $TR --standalone --nproc_per_node=8 src/train/train.py sweep_muon_lr${LR}
done

echo "======== 汇总对比 ========"; date
$PY - <<'EOF'
import csv, os
ADAM = "checkpoints_b64/loss_log.csv"
# Adam 基准：取几个早段步的 val_loss（线性插值到扫描 eval 步）
adam = []
with open(ADAM) as f:
    for r in csv.DictReader(f):
        adam.append((int(r["step"]), float(r["val_loss"])))
def adam_at(step):
    # 线性插值
    for i in range(len(adam)-1):
        s0,v0 = adam[i]; s1,v1 = adam[i+1]
        if s0 <= step <= s1:
            return v0 + (v1-v0)*(step-s0)/max(1,s1-s0)
    return adam[-1][1] if step>=adam[-1][0] else adam[0][1]

print(f"{'lr':>6} {'step':>6} {'muon_val':>9} {'adam_val':>9} {'Δ(muon-adam)':>13}")
rows = {}
for lr in (80,160,240,320):
    p = f"test_outputs/sweep_muon_lr{lr}/loss_log.csv"
    if not os.path.exists(p):
        print(f"{lr:>6}  (无 loss_log，可能 diverge/失败)"); continue
    with open(p) as f:
        L = [(int(r["step"]),float(r["val_loss"])) for r in csv.DictReader(f)]
    rows[lr] = L
    for step,val in L:
        if step==0: continue
        av = adam_at(step)
        mark = "  ← Muon 更好" if val < av else ""
        print(f"{lr:>6} {step:>6} {val:>9.4f} {av:>9.4f} {val-av:>+13.4f}{mark}")
    print()
# 每个 lr 的末端 val
print("=== 各 lr 末端 val_loss（越低越好；Adam 同 step 见上）===")
for lr in sorted(rows):
    L = rows[lr]
    last_step, last_val = L[-1]
    print(f"  lr={lr:>4}: step{last_step} val={last_val:.4f} vs adam={adam_at(last_step):.4f}")
EOF
date
