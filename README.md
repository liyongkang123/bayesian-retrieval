# Bayesian Neural Information Retrieval (BNIR)

## Installation

Create a conda environment, e.g., Python version 3.10. (pyserini need)
```bash
conda create --name bnir python=3.10
```

Activate the environment and install PyTorch, e.g., for CUDA 11.8.
```bash
conda activate bnir
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install custom tevatron fork (editable flag `-e` is optional).
```bash
cd $PROJECT_ROOT
git clone git@github.com:peustr/tevatron.git
cd tevatron
pip install -e .
```

Install BNIR and its requirements (editable flag `-e` is optional).
```bash
cd $PROJECT_ROOT
git clone git@github.com:peustr/bayesian-retrieval.git
cd bayesian-retrieval
pip install -r requirements.txt
pip install -e .
```

## Experiments

### BERT baseline

Train a model:
```
sbatch scripts/baseline/train_bert_msmarco.sh
```

Encode the corpus (requires Hugging Face access token):
```
sbatch scripts/baseline/encode_msmarco.sh
```

Evaluate the trained model (requires Java 11 installation):
```
sbatch scripts/baseline/search_msmarco.sh
```

Performance:
```
MRR@10: 0.323
```

### Bayesian BERT

Train a model:
```
sbatch scripts/vi/train_bert_msmarco.sh
```

TODO: Rest of scripts.
