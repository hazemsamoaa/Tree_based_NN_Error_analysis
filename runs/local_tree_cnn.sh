#!/bin/env bash

export CUDA_LAUNCH_BLOCKING=1
module load Java/11.0.20

DATA_MODE=OssBuilds
TRAIN_MODE=tree_cnn
OUTPUT_DIR=./checkpoints/${DATA_MODE}/${TRAIN_MODE}
DATA_DIR="./data/data/${DATA_MODE}/"
NODE_MAP_PATH=./checkpoints/tree_cnn_emb/node_map.bin
EMBEDDINGS_PATH=./checkpoints/tree_cnn_emb/embeddings.bin


which python
# SEEDS=(19 42 123 2023 777)
SEEDS=(19)

for seed in "${SEEDS[@]}"; do
    echo "Running ${TRAIN_MODE} training with seed=${seed}"
    python ./src/train.py \
        --train_data_dir ${DATA_DIR} \
        --seed ${seed} \
        --epochs 10 \
        --repr_epochs 10 \
        --test_size 0.2 \
        --lr 1e-4 \
        --batch_size 4 \
        --train_on ${TRAIN_MODE} \
        --output_dir "${OUTPUT_DIR}/seed_${seed}/" \
        --minsize -1 \
        --maxsize -1 \
        --limit -1 \
        --per_node -1 \
        --num_feats 100 \
        --hidden_size 300  \
        --num_conv 1 \
        --conv_hidden_size 100 \
        --node_map_path $NODE_MAP_PATH \
        --embeddings_path $EMBEDDINGS_PATH 
done

echo "Calculating the average over the seeds=${SEEDS[@]}"
python src/report_over_seed.py --output_dir ${OUTPUT_DIR}/