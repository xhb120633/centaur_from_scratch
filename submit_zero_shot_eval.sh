#!/bin/bash
#SBATCH --job-name=zero_shot_centaur_eval        # Job name
#SBATCH --output=zero_shot_centaur_eval_%j.out   # Standard output log (%j is job ID)
#SBATCH --error=zero_shot_centaur_eval_%j.err    # Standard error log (%j is job ID)
#SBATCH --account=gts-rwilson337-postpaid        # Account name
#SBATCH --nodes=1                                # Request 1 node
#SBATCH --gres=gpu:H200:1                        # Request 1 H200 GPU (141GB is perfect for 70B)
#SBATCH --mem-per-gpu=64G                       # Memory allocated per GPU
#SBATCH --cpus-per-task=8                       # Number of CPU cores per task
#SBATCH --time=8:00:00                           # Time limit: 8 hours (should be enough for direct inference)
#SBATCH --tmp=256G                               # Temporary disk storage
#SBATCH --mail-type=BEGIN,END,FAIL               # Email notifications for job events
#SBATCH --mail-user=hanboxie1997@gatech.edu

# Load Anaconda module (adjust the module name if needed)
module load anaconda3/2023.03
source $(conda info --base)/etc/profile.d/conda.sh
conda activate centaur_eval  # Activate your environment

# Change to the project directory
cd /storage/coda1/p-rwilson337/0/hxie88/centaur_from_scratch

# Set environment variables
export HF_HOME=/storage/scratch1/2/hxie88/hf_cache
export TORCH_COMPILE_CACHE_DIR=/storage/scratch1/2/hxie88/torch_compile_cache
export TORCH_HOME=/storage/scratch1/2/hxie88/torch_cache
export TRITON_CACHE_DIR=/storage/scratch1/2/hxie88/triton_cache
export CUDA_VISIBLE_DEVICES=0

# Set HuggingFace environment variables
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

# Set PyTorch environment variables for optimal performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0

echo "Starting Zero-Shot Centaur Evaluation (Refactored Pipeline)..."
echo "Model: marcelbinz/Llama-3.1-Centaur-70B-adapter"
echo "Task: dubois2022value"
echo "Pipeline: Refactored evaluation with shared core functions"
echo "Packages: unsloth + transformers + trl (same as original)"
echo "Difference: Only prompts changed to zero-shot (no behavioral history)"
echo "Expected Runtime: Similar to original evaluation (~1-2 hours)"

# Launch the zero-shot evaluation script with optimized settings
srun python evaluate_zero_shot_centaur.py \
    --task hilbig2014generalized \
    --model marcelbinz/Llama-3.1-Centaur-70B-adapter \
    --batch-size 4 \
    --skip-detailed-analysis
