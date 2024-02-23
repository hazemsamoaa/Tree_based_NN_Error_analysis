#!/bin/env bash

export CUDA_LAUNCH_BLOCKING=1

OUTPUT_DIR=./checkpoints/oss_builds
DATA_DIR=./data/data/OssBuilds/

# source /mimer/NOBACKUP/groups/snic2022-22-883/APP/my_python/venv/bin/activate
which python
# echo "IT IS STARTING ... "

# 42, 101, 1,
python src/train.py \
    --data_dir $DATA_DIR \
    --seed 42 \
    --epochs 10 \
    --test_size 0.2 \
    --lr 1e-4 \
    --batch_size 4 \
    --train_on dou_transformer \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 2048 \
    --d_model 768 \
    --n_head 8 \
    --d_ff 2048 \
    --n_layer 6 \
    --drop 0.1 \
    --do_log
