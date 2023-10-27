#!/bin/sh
#SBATCH --job-name=search_msmarco
#SBATCH --partition gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x-%j.out

echo "Searching corpus..."

python -m tevatron.faiss_retriever \
  --query_reps query_emb.pkl \
  --passage_reps 'corpus_emb.*.pkl' \
  --depth 100 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to rank.txt

echo "Converting results to MS MARCO format..."

python -m tevatron.utils.format.convert_result_to_marco \
  --input rank.txt \
  --output rank.txt.marco

echo "Generating evaluation metrics..."

python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset rank.txt.marco
