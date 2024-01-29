#!/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=naiss2023-22-354
#SBATCH --partition=alvis
#SBATCH --time=0-12:00:00
#SBATCH --job-name=runner
#SBATCH --error=logs/runner-%J.err.log
#SBATCH --output=logs/runner-%J.out.log
#SBATCH --gpus-per-node=A40:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=samoaa@chalmers.se


# export GDOWN_CACHE_DIR="/mimer/NOBACKUP/groups/snic2022-22-883/.cache/gdown"
# export PIP_CACHE_DIR="/mimer/NOBACKUP/groups/snic2022-22-883/.cache/pip"
# export HF_DATASETS_CACHE="/mimer/NOBACKUP/groups/snic2022-22-883/.cache/huggingface/datasets"
# export TRANSFORMERS_CACHE="/mimer/NOBACKUP/groups/snic2022-22-883/.cache/huggingface/transformers"

export CUDA_LAUNCH_BLOCKING=1


module purge
module load git-lfs/3.2.0
module load CUDA/11.1.1-GCC-10.2.0
module load Python/3.8.6-GCCcore-10.2.0
module load Java/11.0.20

git config --global user.name "Peter Samoaa"
git config --global user.email "hazim.samoaa@gmail.com"
git config --global credential.helper store

OUTPUT_DIR=
DATA_DIR=

source /mimer/NOBACKUP/groups/snic2022-22-883/APP/my_python/venv/bin/activate
which python
echo "IT IS STARTING ... "

python src/train.py \
    --data_dir $DATA_DIR \
    --seed 42 \
    --epochs 10 \
    --repr_epochs 10 \
    --test_size 0.2 \
    --lr 1e-3 \
    --batch_size 8 \
    --train_on tree_cnn \
    --output_dir $OUTPUT_DIR \
    --minsize -1 \
    --maxsize -1 \
    --limit -1 \
    --per_node -1 \
    --num_feats 100 \
    --hidden_size 300