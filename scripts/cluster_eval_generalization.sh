#!/bin/bash

#SBATCH -J CENTaUR2
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 24:00:00
#SBATCH --constraint=a100_80gb
#SBATCH --nice=10000
#SBATCH --cpus-per-task=20

cd ../generalization/

source activate unsloth_env2

python generalization.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter
