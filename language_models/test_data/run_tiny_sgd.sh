#!/bin/bash
# SGD run: same tiny transformer / dataset, but SGD(momentum=0.9) with a much
# larger LR than Adam (SGD needs 0.1~1.0 range on this task; 6e-4 would look
# like "SGD can't learn"). Separate out/log/comment so it doesn't clobber Adam.
python -u train_gpt2.py config/train_gpt2_tiny.py \
    --dataset=synth_uniform_balanced \
    --use_sgd=True \
    --learning_rate=0.3 \
    --comment='tiny_sgd' \
    --save_dir='log_gpt2/tiny_sgd' \
    --out_dir='out-gpt2/tiny_sgd'
