#!/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=naiss2023-22-354
#SBATCH --partition=alvis
#SBATCH --time=0-12:00:00
#SBATCH --job-name=tree-emb
#SBATCH --error=logs/runner-%J.err.log
#SBATCH --output=logs/runner-%J.out.log
#SBATCH --gpus-per-node=A100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=samoaa@chalmers.se

export CUDA_LAUNCH_BLOCKING=1

module purge
module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1
module load Java/11.0.20

source /mimer/NOBACKUP/groups/snic2022-22-883/APP/python_app/venv/bin/activate

TRAIN_MODE=tree_cnn_emb
OUTPUT_DIR=./checkpoints/${TRAIN_MODE}/
DATA_DIR="./data/data/HadoopTests/ ./data/data/OssBuilds/"

# SEEDS=(19 42 123 2023 777)
SEEDS=(19)
which python
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
        --output_dir ${OUTPUT_DIR} \
        --minsize -1 \
        --maxsize -1 \
        --limit -1 \
        --per_node -1 \
        --num_feats 100 \
        --hidden_size 300 
done
