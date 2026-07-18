#!/bin/bash
# Adam run: tiny transformer on synth_uniform_balanced, weight tying disabled.
# Run from language_models/test_data/ so relative paths (config/, ../../data..., files/) resolve.
python -u train_gpt2.py config/train_gpt2_tiny.py \
    --dataset=synth_uniform_balanced \
    --use_sgd=False \
    --learning_rate=6e-4 \
    --comment='tiny_adam' \
    --save_dir='log_gpt2/tiny_adam' \
    --out_dir='out-gpt2/tiny_adam'
