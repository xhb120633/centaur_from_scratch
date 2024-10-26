#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

module load PyTorch/2.1.2

python $PROJECT_hai_centaur2/Llama-3.1-Centaur-70B/test.py --model marcelbinz/Llama-3.1-Centaur-70B
