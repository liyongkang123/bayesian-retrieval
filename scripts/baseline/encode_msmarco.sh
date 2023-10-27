#!/bin/sh
#SBATCH --job-name=encode_msmarco
#SBATCH --partition gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x-%j.out

echo "Encoding corpus..."

for s in $(seq -f "%02g" 0 19)
do
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path models \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --p_max_len 128 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encoded_save_path corpus_emb.${s}.pkl \
  --encode_num_shard 20 \
  --encode_shard_index ${s}
done

echo "Encoding queries..."

python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path models \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/msmarco-passage/dev \
  --encoded_save_path query_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry
