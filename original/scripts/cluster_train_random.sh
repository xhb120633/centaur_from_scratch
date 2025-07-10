#!/bin/bash
# Training from random weights for specialized choice prediction task
# Goal: Learn experimental context -> human choice mapping, not general language modeling

#SBATCH -J CENTaUR-Random
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH -t 48:00:00
#SBATCH --constraint=a100_80gb
#SBATCH --nice=10000
#SBATCH --cpus-per-task=20

source activate unsloth_env2

python finetune_random.py \
--seed 100 \
--model_name_or_path "unsloth/Meta-Llama-3.1-70B-bnb-4bit" \
--max_seq_len 32768 \
--num_train_epochs 25 \
--log_level "info" \
--logging_strategy "steps" \
--logging_steps 1 \
--evaluation_strategy "steps" \
--eval_steps 500 \
--save_strategy "steps" \
--save_steps 500 \
--learning_rate 1e-4 \
--optim "adamw_8bit" \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--warmup_steps 1000 \
--output_dir "centaur-random-init" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 32 \
--random_init true \
--load_best_model_at_end true \
--metric_for_best_model "eval_loss" \
--greater_is_better false \
--save_total_limit 3 