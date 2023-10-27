#!/bin/sh
#SBATCH --job-name=train_bert_msmarco
#SBATCH --partition gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x-%j.out

python -m bnir.runner \
  --output_dir vi_models \
  --model_name_or_path bert-base-uncased \
  --model_type vi \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_n_passages 8 \
  --learning_rate 5e-6 \
  --q_max_len 16 \
  --p_max_len 128 \
  --num_train_epochs 3 \
  --logging_steps 500 \
  --overwrite_output_dir
