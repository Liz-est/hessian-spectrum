#!/bin/bash

# 1. Activate conda environment
source /libingheng/miniconda3/etc/profile.d/conda.sh
conda activate cvmodels  # replace with your conda environment name

# 2. Change directory to specific path
cd /libingheng/hessian-spectrum/vision_models/  # replace with your actual path

export CUDA_VISIBLE_DEVICES=0

# 3. Define lists for load_iter and dataset_mode
load_iters=(1 11 41 61 89)  # modify this list as needed
dataset_modes=("uniform")  # dataset_mode options
arch=("vit_base")

# 4. Nested loop through load_iter and dataset_mode
for arch in "${arch[@]}"
do
    for mode in "${dataset_modes[@]}"
    do
        for iter in "${load_iters[@]}"
        do
            echo "Running with dataset_mode=$mode, load_iter=$iter"
            python -um main \
              --data '/libingheng/dataset/imagenet' \
              --dataset_mode "$mode" \
              --opt 'adamw' \
              --workers 6 \
              --arch "$arch" \
              --seed 42 \
              --batchsize 128 \
              --resume "/libingheng/hessian-spectrum/vision_models/checkpoint/vit_normal_init/" \
              --load_iter "$iter" \
              --comment "" \
              --use_minibatch \
              --gradient_accumulation_steps 1 \
              --plot_hessian
        done
    done
done