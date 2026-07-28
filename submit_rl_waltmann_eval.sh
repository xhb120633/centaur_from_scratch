#!/bin/bash
#SBATCH --job-name=rl_waltmann_eval
#SBATCH --output=rl_waltmann_eval_%j.out
#SBATCH --error=rl_waltmann_eval_%j.err
#SBATCH --account=gts-rwilson337-postpaid
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --tmp=256G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hanboxie1997@gatech.edu

set -euo pipefail

module load anaconda3/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /storage/project/r-rwilson337-0/hxie88/conda_envs/centaur_eval

cd /storage/project/r-rwilson337-0/hxie88/centaur_from_scratch

export HF_HOME=/storage/scratch1/2/hxie88/hf_cache
export TORCH_COMPILE_CACHE_DIR=/storage/scratch1/2/hxie88/torch_compile_cache
export TORCH_HOME=/storage/scratch1/2/hxie88/torch_cache
export TRITON_CACHE_DIR=/storage/scratch1/2/hxie88/triton_cache
export CUDA_VISIBLE_DEVICES=0

export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0

DATASET=rl_waltmann_centaur

printf 'Starting %s full-context + context-only Centaur evaluation\n' "$DATASET"
printf 'Project: %s\n' "$(pwd)"
printf 'Source CSV: rl_waltmann_centaur.csv\n'
printf 'Dataset JSONL: datasets/main_test_tasks/%s.jsonl\n' "$DATASET"
printf 'Model: marcelbinz/Llama-3.1-Centaur-70B-adapter\n'

python -B generate_rl_jsonl.py \
    --input-csv rl_waltmann_centaur.csv \
    --output-jsonl datasets/main_test_tasks/rl_waltmann_centaur.jsonl \
    --experiment-tag rl_waltmann/centaur.csv

srun python -B evaluate_history_only_centaur.py \
    --task rl_waltmann_centaur \
    --model marcelbinz/Llama-3.1-Centaur-70B-adapter \
    --batch-size 2 \
    --skip-detailed-analysis \
    --choice-nll-agg average \
    --prompt-mode both
