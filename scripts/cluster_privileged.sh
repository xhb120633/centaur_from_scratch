#!/bin/bash

#SBATCH -J privileged
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH --nice=1000
#SBATCH --cpus-per-task=32

source activate unsloth_env2

cd ../generalization/
python privileged.py
