#!/bin/bash
# Full pipeline for vanilla_transformer: train (loss curve + 4 checkpoints) then analyze
# (Hessian spectra, last-layer + hidden-neuron hetero, evolution plots).
#
#   bash run_C.sh          # full run (~1h on CPU)
#
# Outputs:
#   runs/vanilla_transformer/loss_curve.png, ckpt_{init,p10,p50,p100}.pt
#   files/vanilla_transformer/<tag>/spectrum_*.png, hetero_*_{skl,js}.png
#   files/vanilla_transformer/evolution_{skl,js}.png, all_summary.json
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8

echo "=== [1/2] training ==="
python3 -u train_vanilla_transformer.py "$@"

echo "=== [2/2] analyzing checkpoints ==="
python3 -u analyze_vanilla.py

echo "=== done ==="
