# ePQA NLP Modeling

This repository contains the implementation for:
- **Part I:** word-level language modeling on ePQA-style inputs (n-gram / RNN / LSTM / decoder-only Transformer)
- **Part II:** embedding ablations (scratch vs self-trained vs public pretrained; fixed vs fine-tuned)
- **Part III:** downstream **3-way candidate quality classification** using LM representations (pooling + partial fine-tuning)
  (Please take after the paths of files!!)
## Repository layout

- `data_cleaning.py` — cleans/normalizes raw candidate texts and builds LM-ready corpora.
- `lm_training.py` — trains and evaluates language models; saves checkpoints and logs; supports qualitative generation.
- `embedding/` — scripts for Part II embedding variants and alignment utilities.
- `downstream task/` — scripts for Part III downstream classification and fine-tuning settings.
- `samples_sample.txt` — example generation outputs.

## Environment

### Hardware
- GPU: NVIDIA GeForce RTX 4060 Ti (16GB VRAM)
- CPU: Intel Core i5-12600KF
- RAM: 32GB

### Software
- Python 3.x
- PyTorch
- Common utilities: numpy, pandas, tqdm (and any other packages required by the scripts)

Install dependencies (example):
```bash
pip install torch numpy pandas tqdm
````

## Data

The scripts are designed for the ePQA assignment setting. In general, they expect dataset splits in **CSV** form (e.g., `train/dev/test`) with fields needed to build:

* LM input text like `question: ... candidate: ... answer: ...` (Part I/II)
* (question, candidate) pairs + 3-way labels (Part III)

Because column names can differ across versions of the dataset, please use:

```bash
python data_cleaning.py -h
python lm_training.py -h
```

to see the exact required/optional arguments and expected columns.

## Quickstart

### 1) (Optional) Clean / preprocess

Run the cleaning script to produce normalized text suitable for LM training:

```bash
python data_cleaning.py \
  --input <PATH_TO_RAW_SPLIT_OR_FILE> \
  --outdir processed_data
```

### 2) Part I: Train language models

Train and evaluate the models (perplexity + qualitative generation):

```bash
python lm_training.py \
  --data <PATH_TO_PROCESSED_DATA> \
  --outdir runs/part1 \
  --device cuda
```

### 3) Part II: Embedding variants

Part II scripts are under `embedding/`. Typical workflow:

1. prepare / align embeddings to the vocabulary
2. run LM training with different embedding modes (fixed vs fine-tune)

```bash
cd embedding
# See available scripts and options
python <script_name>.py -h
```

### 4) Part III: Downstream classification

Part III scripts are under `downstream task/`. The task is **3-way classification**:

* `0`: irrelevant / not helpful candidate
* `1`: partially helpful / incomplete candidate
* `2`: fully answer-supporting candidate

```bash
cd "downstream task"
python <script_name>.py -h
```

## Notes on reproducibility

* For fair comparisons, I keep preprocessing and evaluation consistent across settings.
* When models/budgets cannot be perfectly matched, I document the differences and keep comparisons reasonable (e.g., same data splits, same vocabulary, same evaluation protocol).

## Outputs

Depending on the script, outputs may include:

* training logs (loss curves, dev metrics)
* saved vocab / tokenizer artifacts
* checkpoints (`.pt`)
* evaluation summaries (e.g., perplexity, F1/Macro-F1 for classification)
* qualitative generation samples

## License

MIT License (see `LICENSE`).

```
::contentReference[oaicite:0]{index=0}
```



