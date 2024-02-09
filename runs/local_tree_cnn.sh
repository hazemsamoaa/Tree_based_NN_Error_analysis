#!/bin/env bash

export CUDA_LAUNCH_BLOCKING=1


# module purge
# module load git-lfs/3.2.0
# module load CUDA/11.1.1-GCC-10.2.0
# module load Python/3.8.6-GCCcore-10.2.0
# module load Java/11.0.20

# git config --global user.name "Peter Samoaa"
# git config --global user.email "hazim.samoaa@gmail.com"
# git config --global credential.helper store

OUTPUT_DIR=./checkpoints/
DATA_DIR=./data/data/HadoopTests/
NODE_MAP_PATH=./checkpoints/tree_cnn_emb/node_map.bin
EMBEDDINGS_PATH=./checkpoints/tree_cnn_emb/embeddings.bin

# source /mimer/NOBACKUP/groups/snic2022-22-883/APP/my_python/venv/bin/activate
which python
# echo "IT IS STARTING ... "

# 42, 101, 1, 
python src/train.py \
    --data_dir $DATA_DIR \
    --seed 42 \
    --epochs 10 \
    --repr_epochs 10 \
    --test_size 0.2 \
    --lr 1e-4 \
    --batch_size 4 \
    --train_on tree_cnn \
    --output_dir $OUTPUT_DIR \
    --minsize -1 \
    --maxsize -1 \
    --limit -1 \
    --per_node -1 \
    --num_feats 100 \
    --hidden_size 300 \
    --num_conv 1 \
    --conv_hidden_size 100 \
    --node_map_path $NODE_MAP_PATH \
    --embeddings_path $EMBEDDINGS_PATH 