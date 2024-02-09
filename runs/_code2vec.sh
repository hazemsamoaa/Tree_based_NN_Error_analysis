#!/bin/env bash

module load Java/11.0.20

OUTPUT_DIR=./checkpoints
DATA_DIR=./data/OssBuilds

python src/train.py \
    --data_dir $DATA_DIR \
    --seed 42 \
    --epochs 10 \
    --repr_epochs 10 \
    --test_size 0.2 \
    --lr 1e-3 \
    --batch_size 8 \
    --train_on code2vec \
    --output_dir $OUTPUT_DIR \
    --predict \
    --export_code_vectors \
    --load ./data/models/java14_model/saved_model_iter8.release