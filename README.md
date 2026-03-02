# ePQA NLP Modeling

This repository contains the implementation for:
- **Part I:** word-level language modeling on ePQA-style inputs (n-gram / RNN / LSTM / decoder-only Transformer)
- **Part II:** embedding ablations (scratch vs self-trained vs public pretrained; fixed vs fine-tuned)
- **Part III:** downstream **3-way candidate quality classification** using LM representations (pooling + partial fine-tuning)
  (Please take after the paths of files!!)


## Results Summary 
### Part I — Language Modeling (PPL + Efficiency)
**Task:** train word-level LMs on concatenated sequences `question: ... candidate: ... answer: ...` and generate the answer continuation.

| Model | Dev PPL | Test PPL | Train Time (s) | Sec/Forward | Tokens/Sec |
|---|---:|---:|---:|---:|---:|
| n-gram | 496.78 | 464.44 | 2.63 | 0.008010 | 783,510 |
| RNN | 47.71 | 45.29 | 1,015.31 | 0.023407 | 349,983 |
| LSTM | 36.67 | 35.52 | 1,032.27 | 0.014607 | 560,812 |
| Transformer | 17.96 | 17.42 | 1,501.80 | 0.024419 | 335,472 |

**Conclusion:** Transformer achieves the best perplexity (strongest modeling capacity), while LSTM offers the best throughput under the chosen batch/sequence settings; n-gram trains fast but performs poorly on this conditional format.

---

### Part II — Embedding Ablations (Fixed vs Fine-tuned; Self-trained vs Public)
**Setup:** same LM architectures/training as Part I; only change embedding initialization and whether embeddings are updated during LM training. Reported as mean ± std over seeds {1234, 2002, 2026}. Public embeddings are aligned to the 30k vocab (coverage shown).

**Best setting per LM (Dev/Test PPL):**

| LM | Best embedding setting | Dev PPL | Test PPL | Notes |
|---|---|---:|---:|---|
| RNN | Public-Finetune **GloVe** | 41.97 ± 0.39 | 39.96 ± 0.37 | coverage ≈ 0.449 |
| LSTM | **Self-Fixed Word2Vec** (Dev-best) / Public-Finetune GloVe (Test-best) | 32.58 ± 0.86 / 32.61 ± 0.26 | 31.60 ± 0.75 / **31.51 ± 0.10** | Word2Vec coverage ≈ 1.000 |
| Transformer | Public-Finetune **GloVe** | **17.20 ± 0.07** | **16.84 ± 0.09** | coverage ≈ 0.449 |

**Key conclusions:**
- Fixed **self-trained Word2Vec** embeddings are consistently strong for RNN/LSTM (on-domain signal + full coverage).
- **Public-Fixed** can hurt badly because coverage is only ~45–50%; missing/OOV tokens remain poorly represented when frozen (especially harmful for Transformer).
- **Public-Finetune** largely resolves the mismatch and recovers/improves performance; finetuned GloVe is the best overall for Transformer.

Optional figure (add the PNG to your repo, then update the path):
- `assets/part2_training_stability_devppl_row_meanstd_seeds1234_2002_2026.png`

```text
![Part II stability curves](assets/part2_training_stability_devppl_row_meanstd_seeds1234_2002_2026.png)
````

---

### Part III — Downstream Task (3-way Candidate Quality Classification)

**Task:** classify `question: ... candidate: ...` into:

* **0**: irrelevant / not helpful
* **1**: partially helpful / incomplete
* **2**: fully answer-supporting

#### Cross-LM transfer (mean pooling; Macro-F1 is the main metric)

| LM          | LM setting | Dev F1 | Dev Acc | Test F1 | Test Acc | Train (s) |
| ----------- | ---------- | -----: | ------: | ------: | -------: | --------: |
| RNN         | Freeze     | 0.4103 |  0.6157 |  0.3743 |   0.6285 |     148.1 |
| RNN         | Tune all   | 0.4430 |  0.6745 |  0.4128 |   0.6831 |     279.2 |
| LSTM        | Freeze     | 0.3896 |  0.6227 |  0.3683 |   0.6318 |     178.7 |
| LSTM        | Tune all   | 0.5011 |  0.6540 |  0.4779 |   0.6615 |     326.8 |
| Transformer | Freeze     | 0.4862 |  0.6734 |  0.4589 |   0.6741 |     308.2 |
| Transformer | Tune all   | 0.5441 |  0.7226 |  0.5255 |   0.7185 |     735.3 |

**Conclusion:** Transformer representations transfer best; fine-tuning improves all models but increases compute.

#### Transformer ablations (pooling + fine-tuning scope)

| Representation | LM setting  | Dev F1 | Dev Acc |    Test F1 |   Test Acc | Train (s) |
| -------------- | ----------- | -----: | ------: | ---------: | ---------: | --------: |
| Mean pooling   | Freeze      | 0.4782 |  0.5724 |     0.4761 |     0.5803 |     196.4 |
| EOS pooling    | Freeze      | 0.4934 |  0.6238 |     0.4766 |     0.6249 |     408.4 |
| Mean pooling   | Tune all    | 0.5652 |  0.7325 |     0.5395 | **0.7188** |     720.9 |
| Mean pooling   | Tune last-2 | 0.5732 |  0.7118 | **0.5632** |     0.7075 |     447.1 |

**Key conclusions:**

* **Mean pooling** works better than EOS pooling once adaptation is allowed (more robust aggregation across tokens).
* **Partial fine-tuning** (tuning only the last 1–2 Transformer blocks) yields the best Macro-F1 with a better compute–performance trade-off than tuning all layers.
* Errors are concentrated on the middle label (**class 1**), with confusion between classes 1 and 2.

Optional figure (add the PNG to your repo, then update the path):

* `assets/part3_best_confusion.png`

```text
![Part III confusion matrix](assets/part3_best_confusion.png)
```

```
::contentReference[oaicite:0]{index=0}
```




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



