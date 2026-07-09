#!/bin/bash

# 1. Activate conda environment
source /libingheng/miniconda3/etc/profile.d/conda.sh
conda activate cvmodels  # replace with your conda environment name

# 2. Change directory to specific path
cd /libingheng/hessian-spectrum/vision_models/  # replace with your actual path

# 3. Add wandb API key
export SWANLAB_API_KEY=7Jj19kfBocdDpgKfDdzNh
export SWANLAB_DIR=./swanlab  # 例如: export SWANLAB_DIR=$HOME/.swanlab

python -um main \
  --data '/libingheng/dataset/imagenet'\
  --dataset_mode 'uniform' \
  --lr 1e-3\
  --warmup_epochs 5 \
  --beta1 0.9\
  --beta2  0.9\
  --wd 1e-4\
  --workers 6\
  --epochs 90\
  --arch 'resnet18'\
  --opt 'adamw'\
  --seed 32 \
  --batchsize 1024\
  --epsilon 1e-8\
  --resume ''\
  --load_iter 0\
  --comment 'val_attn' \

