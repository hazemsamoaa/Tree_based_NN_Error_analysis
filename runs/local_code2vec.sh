#!/bin/env bash

export CUDA_LAUNCH_BLOCKING=1
module load Java/11.0.20

DATA_MODE=OssBuilds
TRAIN_MODE=code2vec
OUTPUT_DIR=./checkpoints/${DATA_MODE}/${TRAIN_MODE}
DATA_DIR="./data/data/${DATA_MODE}/"

which python

# SEEDS=(19 42 123 2023 777)
SEEDS=(19)

for seed in "${SEEDS[@]}"; do
    echo "Running training with seed=$seed"
    python src/train.py \
        --train_data_dir ${DATA_DIR} \
        --seed ${seed} \
        --epochs 10 \
        --test_size 0.2 \
        --lr 1e-4 \
        --batch_size 4 \
        --train_on "${TRAIN_MODE}" \
        --output_dir "${OUTPUT_DIR}/seed_${seed}/" \
        --predict \
        --export_code_vectors \
        --load ./data/models/java14_model/saved_model_iter8.release \
        --jar_path ./scripts/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar \
        --do_log
done

echo "Calculating the average over the seeds=${SEEDS[@]}"
python src/report_over_seed.py --output_dir ${OUTPUT_DIR}/