#!/bin/bash
#SBATCH --job-name=predictive_rl_latest_eval
#SBATCH --output=predictive_rl_latest_eval_%j.out
#SBATCH --error=predictive_rl_latest_eval_%j.err
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

echo "Starting latest predictive_rl full-context + context-only Centaur evaluation"
echo "Project: $(pwd)"
echo "Source CSV: all_simulations_summary_centaur.csv"
echo "Dataset JSONL: datasets/main_test_tasks/predictive_rl_exp1.jsonl"
echo "Model: marcelbinz/Llama-3.1-Centaur-70B-adapter"

python -B generate_rl_jsonl.py \
    --input-csv all_simulations_summary_centaur.csv \
    --output-jsonl datasets/main_test_tasks/predictive_rl_exp1.jsonl \
    --experiment-tag predictive_rl/exp1.csv

srun python -B evaluate_history_only_centaur.py \
    --task predictive_rl_exp1 \
    --model marcelbinz/Llama-3.1-Centaur-70B-adapter \
    --batch-size 2 \
    --skip-detailed-analysis \
    --choice-nll-agg average \
    --prompt-mode both
