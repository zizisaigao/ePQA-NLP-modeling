#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST5230 Assignment 1 - Part I (baseline code)
Train and compare: n-gram / RNN / LSTM / Transformer language models on cleaned ePQA candidate corpus.

Input:
  - cleaned_candidates.parquet (recommended): must include columns:
      text_clean, source, optionally ASIN, qid
  - or corpus.jsonl with fields: {"text": "...", "source": "..."} (no ASIN/qid split possible)

Key features:
  - Split by ASIN (preferred) or random split if ASIN unavailable
  - Word-level tokenizer (whitespace + basic normalization)
  - Vocab build from train only
  - N-gram with add-k smoothing
  - Neural LMs in PyTorch with PPL evaluation
  - Generation samples for qualitative comparison

Run examples:
  # Using parquet and ASIN split
  python part1_train_lms.py --data processed_data/cleaned_candidates.parquet --outdir runs/part1 \
      --split_by asin --vocab_size 30000 --seq_len 128 --batch_size 64 --device cuda

  # Using jsonl (random split)
  python part1_train_lms.py --data processed_data/corpus.jsonl --outdir runs/part1 \
      --vocab_size 30000 --seq_len 128 --batch_size 64 --device cuda

Notes:
  - Ensure your cqa is normalized to "Q: ... <NL> A: ..." and <NL> is kept as a token.
  - If your corpus.txt contains literal "\\n", do NOT use it; use parquet or jsonl.

Dependencies:
  pip install torch pandas pyarrow tqdm
"""

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Tokenizer (word-level)
# -----------------------------
RE_MULTI_SPACE = re.compile(r"\s+")

def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer.
    Assumes <NL> already appears in text as a token separator.
    """
    text = text.strip().lower()
    text = RE_MULTI_SPACE.sub(" ", text)
    if not text:
        return []
    return text.split(" ")


# -----------------------------
# Vocab
# -----------------------------
@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad: int
    unk: int
    bos: int
    eos: int

    @classmethod
    def build(cls, token_lists: List[List[str]], vocab_size: int):
        special = ["<pad>", "<unk>", "<bos>", "<eos>"]
        counter = Counter()
        for toks in token_lists:
            counter.update(toks)
        # reserve special
        most_common = [t for t, _ in counter.most_common(max(0, vocab_size - len(special)))]
        itos = special + most_common
        stoi = {t: i for i, t in enumerate(itos)}
        return cls(stoi=stoi, itos=itos, pad=0, unk=1, bos=2, eos=3)

    def encode(self, toks: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk) for t in toks]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids]


# -----------------------------
# Data loading & splitting
# -----------------------------

def ensure_text_clean_from_qa(
    df: pd.DataFrame,
    include_context: bool = True,
    refuse_text: str = "No answer based on the provided candidate.",
    partial_text: str = "The candidate provides helpful information but does not fully answer the question."
) -> pd.DataFrame:
    """Create 'text_clean' for Part I LM training from QA-style columns if needed.

    If df already contains 'text_clean', it is returned unchanged.
    Expected columns (if text_clean absent): question, candidate, optional context, label, optional answer.
    The resulting text is a single training string that includes the conditioning fields plus a target
    segment (label + answer/refusal), so standard LM training can learn to produce label/answer text.
    """
    if "text_clean" in df.columns:
        return df

    required = {"question", "candidate"}
    if not required.issubset(set(df.columns)):
        raise ValueError("Input data must contain either 'text_clean' or columns: question, candidate (and optionally context, label, answer).")

    def _row_to_text(r) -> str:
        q = str(r.get("question", "")).strip()
        c = str(r.get("candidate", "")).strip()
        ctx = str(r.get("context", "")).strip() if include_context and "context" in df.columns else ""
        lab_i = r.get("label", None)
        try:
            lab_i = int(lab_i) if lab_i is not None and str(lab_i).strip() != "" else None
        except Exception:
            lab_i = None
        ans = str(r.get("answer", "")).strip() if "answer" in df.columns else ""

        parts = [f"Question: {q}", f"Candidate: {c}"]
        if ctx:
            parts.append(f"Context: {ctx}")

        # Target segment (still plain text, so Part I LM setup remains unchanged)
        if lab_i is None:
            #parts.append("Label: ")
            parts.append("Answer: ")
        else:
            #parts.append(f"Label: {lab_i}")
            if lab_i == 2 and ans:
                parts.append(f"Answer: {ans}")
            elif lab_i == 1:
                parts.append(f"Answer: {ans}")  #改了 {partial_text}
            else:
                parts.append(f"Answer: {refuse_text}")

        return "\n".join(parts)

    df = df.copy()
    df["text_clean"] = df.apply(_row_to_text, axis=1)
    if "source" not in df.columns:
        df["source"] = ""
    return df


def load_corpus(path: str, include_context: bool = True) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
        # expect text_clean
        if "text_clean" not in df.columns:
            raise ValueError("Parquet must contain column 'text_clean'.")
        if "source" not in df.columns:
            df["source"] = ""
        return df
    if p.suffix.lower() == ".jsonl":
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows.append({"text_clean": obj.get("text", ""), "source": obj.get("source", "")})
        return pd.DataFrame(rows)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        df = ensure_text_clean_from_qa(df, include_context=include_context)
        return df
    raise ValueError("Unsupported file type. Use .parquet, .jsonl, or .csv")


def load_fixed_splits(
    data_path: str,
    include_context: bool = True
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Load pre-defined train/dev/test splits if they exist.

    Supported layouts:
      1) data_path is a directory containing train.csv, dev.csv, test.csv
      2) data_path points to train.csv and dev.csv/test.csv are in the same directory
    Returns (train_df, dev_df, test_df) or None if not found.
    """
    p = Path(data_path)
    if p.is_dir():
        tr, dv, te = p / "train.csv", p / "dev.csv", p / "test.csv"
        if tr.exists() and dv.exists() and te.exists():
            train_df = ensure_text_clean_from_qa(pd.read_csv(tr), include_context=include_context)
            dev_df = ensure_text_clean_from_qa(pd.read_csv(dv), include_context=include_context)
            test_df = ensure_text_clean_from_qa(pd.read_csv(te), include_context=include_context)
            return train_df, dev_df, test_df
        return None

    if p.suffix.lower() == ".csv" and p.name.lower() == "train.csv":
        dv, te = p.parent / "dev.csv", p.parent / "test.csv"
        if dv.exists() and te.exists():
            train_df = ensure_text_clean_from_qa(pd.read_csv(p), include_context=include_context)
            dev_df = ensure_text_clean_from_qa(pd.read_csv(dv), include_context=include_context)
            test_df = ensure_text_clean_from_qa(pd.read_csv(te), include_context=include_context)
            return train_df, dev_df, test_df
    return None


def split_df(
    df: pd.DataFrame,
    split_by: str = "asin",
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 1234
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    split_by:
      - 'asin': group split by ASIN (requires column ASIN)
      - 'qid' : group split by qid (requires column qid)
      - 'random': random row split
    """
    rng = random.Random(seed)
    split_by = split_by.lower()

    if split_by in ("asin", "ASIN".lower()) and "ASIN" in df.columns:
        keys = list(df["ASIN"].fillna("").astype(str).unique())
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        train_keys = set(keys[:n_train])
        dev_keys = set(keys[n_train:n_train + n_dev])
        test_keys = set(keys[n_train + n_dev:])
        train_df = df[df["ASIN"].astype(str).isin(train_keys)]
        dev_df = df[df["ASIN"].astype(str).isin(dev_keys)]
        test_df = df[df["ASIN"].astype(str).isin(test_keys)]
        return train_df, dev_df, test_df

    if split_by == "qid" and "qid" in df.columns:
        keys = list(df["qid"].fillna("").astype(str).unique())
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        train_keys = set(keys[:n_train])
        dev_keys = set(keys[n_train:n_train + n_dev])
        test_keys = set(keys[n_train + n_dev:])
        train_df = df[df["qid"].astype(str).isin(train_keys)]
        dev_df = df[df["qid"].astype(str).isin(dev_keys)]
        test_df = df[df["qid"].astype(str).isin(test_keys)]
        return train_df, dev_df, test_df

    # fallback random
    idx = list(range(len(df)))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train_idx = idx[:n_train]
    dev_idx = idx[n_train:n_train + n_dev]
    test_idx = idx[n_train + n_dev:]
    return df.iloc[train_idx], df.iloc[dev_idx], df.iloc[test_idx]


def df_to_token_lists(df: pd.DataFrame, text_col: str = "text_clean", add_eos: bool = True) -> List[List[str]]:
    token_lists = []
    for t in df[text_col].fillna("").astype(str).tolist():
        toks = tokenize(t)
        if not toks:
            continue
        if add_eos:
            toks = toks + ["<eos>"]
        token_lists.append(toks)
    return token_lists


# -----------------------------
# N-gram LM (add-k smoothing)
# -----------------------------
class AddKSmoothingNGram:
    def __init__(self, n: int = 3, k: float = 0.1, vocab_size: int = 0):
        self.n = n
        self.k = k
        self.vocab_size = vocab_size
        self.context_counts = defaultdict(int)      # count(context)
        self.ngram_counts = defaultdict(int)        # count(context, word)

    def train(self, sequences: List[List[int]]):
        n = self.n
        for seq in sequences:
            # add BOS n-1
            seq2 = [0] * (n - 1) + seq  # 0 is <pad> placeholder; better use BOS but fine if consistent
            for i in range(n - 1, len(seq2)):
                context = tuple(seq2[i - (n - 1): i])
                w = seq2[i]
                self.context_counts[context] += 1
                self.ngram_counts[(context, w)] += 1

    def log_prob(self, context: Tuple[int, ...], w: int) -> float:
        # add-k: (c(context,w)+k) / (c(context)+k*V)
        c_cw = self.ngram_counts.get((context, w), 0)
        c_c = self.context_counts.get(context, 0)
        num = c_cw + self.k
        den = c_c + self.k * self.vocab_size
        return math.log(num / den)

    def perplexity(self, sequences: List[List[int]]) -> float:
        n = self.n
        total_logprob = 0.0
        total_tokens = 0
        for seq in sequences:
            seq2 = [0] * (n - 1) + seq
            for i in range(n - 1, len(seq2)):
                context = tuple(seq2[i - (n - 1): i])
                w = seq2[i]
                total_logprob += self.log_prob(context, w)
                total_tokens += 1
        avg_nll = - total_logprob / max(1, total_tokens)
        return math.exp(avg_nll)


# -----------------------------
# Neural LM data (batchify like PTB)
# -----------------------------
def flatten_token_lists(token_lists: List[List[int]], bos_id: int, eos_id: int) -> List[int]:
    # Concatenate documents into a stream with <bos> at start of doc (optional)
    stream = []
    for doc in token_lists:
        stream.append(bos_id)
        stream.extend(doc)  # doc already contains <eos>
    return stream

def batchify(data: List[int], batch_size: int, device: torch.device) -> torch.Tensor:
    data = torch.tensor(data, dtype=torch.long)
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)

def get_batch(source: torch.Tensor, i: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # source: [T, B]
    seq_len = min(seq_len, source.size(0) - 1 - i)
    data = source[i:i+seq_len]             # [seq_len, B]
    target = source[i+1:i+1+seq_len]       # [seq_len, B]
    return data, target


# -----------------------------
# Neural LMs
# -----------------------------
class RNNLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x: torch.Tensor, h=None):
        # x: [T, B]
        emb = self.drop(self.emb(x))
        out, h = self.rnn(emb, h)
        out = self.drop(out)
        logits = self.fc(out)  # [T, B, V]
        return logits, h

class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x: torch.Tensor, h=None):
        emb = self.drop(self.emb(x))
        out, h = self.lstm(emb, h)
        out = self.drop(out)
        logits = self.fc(out)
        return logits, h

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 4,
                 dim_ff: int = 1024, dropout: float = 0.2, max_len: int = 2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_ff, dropout=dropout, batch_first=False)
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        # x: [T, B]
        T, B = x.size()
        pos = torch.arange(0, T, device=x.device).unsqueeze(1).expand(T, B)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.dropout(h)
        # causal mask: [T, T] with True meaning masked
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        out = self.tr(h, mask=mask)
        logits = self.fc(out)
        return logits


# -----------------------------
# Training & Evaluation
# -----------------------------
def eval_ppl_neural(model: nn.Module, data: torch.Tensor, seq_len: int, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        if isinstance(model, (RNNLM, LSTMLM)):
            h = None
            for i in range(0, data.size(0) - 1, seq_len):
                x, y = get_batch(data, i, seq_len)
                logits, h = model(x, h)
                # detach hidden to avoid growing graph even though no grad
                if isinstance(h, tuple):
                    h = tuple(t.detach() for t in h)
                else:
                    h = h.detach() if h is not None else None
                loss = criterion(logits.view(-1, logits.size(-1)), y.reshape(-1))
                total_loss += loss.item()
                total_tokens += y.numel()
        else:
            for i in range(0, data.size(0) - 1, seq_len):
                x, y = get_batch(data, i, seq_len)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.reshape(-1))
                total_loss += loss.item()
                total_tokens += y.numel()
    avg_nll = total_loss / max(1, total_tokens)
    return math.exp(avg_nll)

def train_neural(
    model: nn.Module,
    train_data: torch.Tensor,
    dev_data: torch.Tensor,
    seq_len: int,
    epochs: int,
    lr: float,
    clip: float,
    device: torch.device,
    log_every: int = 200
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_dev = float("inf")
    best_state = None

    model.to(device)

    for ep in range(1, epochs + 1):
        model.train()
        start = time.time()
        total_loss = 0.0
        total_tokens = 0
        steps = 0

        if isinstance(model, (RNNLM, LSTMLM)):
            h = None
            for i in tqdm(range(0, train_data.size(0) - 1, seq_len), desc=f"Epoch {ep}"):
                x, y = get_batch(train_data, i, seq_len)
                optimizer.zero_grad()
                logits, h = model(x, h)
                # detach hidden between batches (truncated BPTT)
                if isinstance(h, tuple):
                    h = tuple(t.detach() for t in h)
                else:
                    h = h.detach() if h is not None else None

                loss = criterion(logits.view(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
                steps += 1

                if steps % log_every == 0:
                    cur_nll = total_loss / max(1, total_tokens)
                    print(f"[train] ep={ep} step={steps} nll={cur_nll:.4f} ppl={math.exp(cur_nll):.2f}")

        else:
            for i in tqdm(range(0, train_data.size(0) - 1, seq_len), desc=f"Epoch {ep}"):
                x, y = get_batch(train_data, i, seq_len)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
                steps += 1

                if steps % log_every == 0:
                    cur_nll = total_loss / max(1, total_tokens)
                    print(f"[train] ep={ep} step={steps} nll={cur_nll:.4f} ppl={math.exp(cur_nll):.2f}")

        dev_ppl = eval_ppl_neural(model, dev_data, seq_len, device)
        epoch_time = time.time() - start
        print(f"[dev] ep={ep} ppl={dev_ppl:.2f} time={epoch_time:.1f}s")

        if dev_ppl < best_dev:
            best_dev = dev_ppl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_dev_ppl": best_dev}


# -----------------------------
# Generation (qualitative)
# -----------------------------
@torch.no_grad()
def generate_from_neural(
    model: nn.Module,
    vocab: Vocab,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 1.0,
    top_k: int = 0,
    device: torch.device = torch.device("cpu")
) -> str:
    model.eval()
    toks = ["<bos>"] + tokenize(prompt)
    ids = [vocab.stoi.get(t, vocab.unk) if t != "<bos>" else vocab.bos for t in toks]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)  # [T,1]

    # For RNN/LSTM, keep hidden state; for Transformer, re-run each step (ok for short gen)
    h = None
    for _ in range(max_new_tokens):
        if isinstance(model, (RNNLM, LSTMLM)):
            logits, h = model(x[-1:], h)  # feed last token
            logits = logits[-1, 0] / max(1e-8, temperature)
        else:
            logits = model(x)             # [T,1,V]
            logits = logits[-1, 0] / max(1e-8, temperature)

        if top_k and top_k > 0:
            vals, idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
            probs = torch.softmax(vals, dim=-1)
            next_id = idx[torch.multinomial(probs, num_samples=1)]
        else:
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_id.view(1, 1)], dim=0)

        if next_id.item() == vocab.eos:
            break

    out_toks = vocab.decode(x.squeeze(1).tolist())
    # remove leading <bos>
    out_toks = [t for t in out_toks if t not in ("<bos>", "<pad>")]
    return " ".join(out_toks)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/", help="Data path: parquet/jsonl/csv, or a directory containing train.csv/dev.csv/test.csv")
    ap.add_argument("--include_context", type=int, default=1, help="When building text from QA CSV: 1=include context field, 0=ignore context")
    ap.add_argument("--outdir", default="result_3", help="output directory")
    ap.add_argument("--split_by", default="asin", choices=["asin", "qid", "random"], help="split strategy")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vocab_size", type=int, default=30000)
    ap.add_argument("--seq_len", type=int, default=128) # increased from 128 to 256 for better context modeling
    ap.add_argument("--batch_size", type=int, default=64) # 64 -> 128 for faster training with more memory
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="cuda", help="cpu or cuda")
    # n-gram
    ap.add_argument("--ngram_n", type=int, default=3)
    ap.add_argument("--ngram_k", type=float, default=0.1)
    # rnn/lstm dims
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hid_dim", type=int, default=256)
    ap.add_argument("--rnn_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    # transformer
    ap.add_argument("--tf_layers", type=int, default=4)
    ap.add_argument("--tf_heads", type=int, default=4)
    ap.add_argument("--tf_ff", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=2048)
    # optim
    ap.add_argument("--lr_rnn", type=float, default=1e-3)
    ap.add_argument("--lr_tf", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load + split
    include_context = bool(args.include_context)
    fixed = load_fixed_splits(args.data, include_context=include_context)
    if fixed is not None:
        train_df, dev_df, test_df = fixed
        print(f"[split] using fixed CSV splits: train={len(train_df)} dev={len(dev_df)} test={len(test_df)}")
    else:
        df = load_corpus(args.data, include_context=include_context)
        train_df, dev_df, test_df = split_df(df, split_by=args.split_by, seed=args.seed)
        print(f"[split] train={len(train_df)} dev={len(dev_df)} test={len(test_df)}")

    # Tokenize documents
    train_tok = df_to_token_lists(train_df, add_eos=True)
    dev_tok = df_to_token_lists(dev_df, add_eos=True)
    test_tok = df_to_token_lists(test_df, add_eos=True)

    # Build vocab from train only
    vocab = Vocab.build(train_tok, vocab_size=args.vocab_size)
    with open(outdir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"itos": vocab.itos}, f, ensure_ascii=False)

    # Encode
    train_ids_docs = [vocab.encode(toks) for toks in train_tok]
    dev_ids_docs = [vocab.encode(toks) for toks in dev_tok]
    test_ids_docs = [vocab.encode(toks) for toks in test_tok]

    # --------------------
    # 1) N-gram
    # --------------------
    print("\n=== Training n-gram ===")
    t0 = time.time()
    ng = AddKSmoothingNGram(n=args.ngram_n, k=args.ngram_k, vocab_size=len(vocab.itos))
    ng.train(train_ids_docs)
    train_time = time.time() - t0
    t1 = time.time()
    dev_ppl_ng = ng.perplexity(dev_ids_docs)
    test_ppl_ng = ng.perplexity(test_ids_docs)
    eval_time = time.time() - t1
    print(f"[ngram] n={args.ngram_n} k={args.ngram_k} train_time={train_time:.1f}s eval_time={eval_time:.1f}s "
          f"dev_ppl={dev_ppl_ng:.2f} test_ppl={test_ppl_ng:.2f}")

    # --------------------
    # Prepare neural stream
    # --------------------
    train_stream = flatten_token_lists(train_ids_docs, vocab.bos, vocab.eos)
    dev_stream = flatten_token_lists(dev_ids_docs, vocab.bos, vocab.eos)
    test_stream = flatten_token_lists(test_ids_docs, vocab.bos, vocab.eos)

    train_data = batchify(train_stream, args.batch_size, device)
    dev_data = batchify(dev_stream, args.batch_size, device)
    test_data = batchify(test_stream, args.batch_size, device)

    results = {
        "ngram": {"dev_ppl": dev_ppl_ng, "test_ppl": test_ppl_ng, "train_time_s": train_time, "eval_time_s": eval_time}
    }

    # --------------------
    # 2) RNN
    # --------------------
    print("\n=== Training RNN ===")
    rnn = RNNLM(len(vocab.itos), args.emb_dim, args.hid_dim, args.rnn_layers, args.dropout).to(device)
    t0 = time.time()
    train_neural(rnn, train_data, dev_data, args.seq_len, args.epochs, args.lr_rnn, args.clip, device)
    tr_time = time.time() - t0
    dev_ppl = eval_ppl_neural(rnn, dev_data, args.seq_len, device)
    test_ppl = eval_ppl_neural(rnn, test_data, args.seq_len, device)
    results["rnn"] = {"dev_ppl": dev_ppl, "test_ppl": test_ppl, "train_time_s": tr_time}
    torch.save(rnn.state_dict(), outdir / "rnn.pt")
    print(f"[rnn] dev_ppl={dev_ppl:.2f} test_ppl={test_ppl:.2f} train_time={tr_time:.1f}s")

    # --------------------
    # 3) LSTM
    # --------------------
    print("\n=== Training LSTM ===")
    lstm = LSTMLM(len(vocab.itos), args.emb_dim, args.hid_dim, args.rnn_layers, args.dropout).to(device)
    t0 = time.time()
    train_neural(lstm, train_data, dev_data, args.seq_len, args.epochs, args.lr_rnn, args.clip, device)
    tr_time = time.time() - t0
    dev_ppl = eval_ppl_neural(lstm, dev_data, args.seq_len, device)
    test_ppl = eval_ppl_neural(lstm, test_data, args.seq_len, device)
    results["lstm"] = {"dev_ppl": dev_ppl, "test_ppl": test_ppl, "train_time_s": tr_time}
    torch.save(lstm.state_dict(), outdir / "lstm.pt")
    print(f"[lstm] dev_ppl={dev_ppl:.2f} test_ppl={test_ppl:.2f} train_time={tr_time:.1f}s")

    # --------------------
    # 4) Transformer
    # --------------------
    print("\n=== Training Transformer ===")
    tf = TransformerLM(
        vocab_size=len(vocab.itos),
        d_model=args.emb_dim,
        nhead=args.tf_heads,
        num_layers=args.tf_layers,
        dim_ff=args.tf_ff,
        dropout=args.dropout,
        max_len=args.max_len
    ).to(device)

    t0 = time.time()
    train_neural(tf, train_data, dev_data, args.seq_len, args.epochs, args.lr_tf, args.clip, device)
    tr_time = time.time() - t0
    dev_ppl = eval_ppl_neural(tf, dev_data, args.seq_len, device)
    test_ppl = eval_ppl_neural(tf, test_data, args.seq_len, device)
    results["transformer"] = {"dev_ppl": dev_ppl, "test_ppl": test_ppl, "train_time_s": tr_time}
    torch.save(tf.state_dict(), outdir / "transformer.pt")
    print(f"[tf] dev_ppl={dev_ppl:.2f} test_ppl={test_ppl:.2f} train_time={tr_time:.1f}s")

    # --------------------
    # Save results
    # --------------------
    with open(outdir / "part1_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # --------------------
    # Qualitative generation (neural models)
    # --------------------
    prompts = [
        # Fully-answering style prompt (expect Label: 2 + Answer: ...)
        "question: does it come with batteries? <nl> candidate: it requires two aa batteries and they are not included. <nl> answer:"

        # Irrelevant candidate prompt (expect Label: 0 + refusal-style Answer)
        "question: what is the warranty period? <nl> candidate: this product is available in red and blue colors. <nl> answer:"

        # Helpful-but-not-fully prompt (expect Label: 1 + partial-style Answer)
        "question: is it compatible with iphone? <nl> candidate: it works with ios devices via bluetooth, but no specific iphone models are listed. <nl> answer:"
    ]
    gen_path = outdir / "generations.txt"
    with open(gen_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(f"\n=== PROMPT: {p} ===\n")
            f.write("[RNN]\n" + generate_from_neural(rnn, vocab, p, device=device, top_k=50) + "\n\n")
            f.write("[LSTM]\n" + generate_from_neural(lstm, vocab, p, device=device, top_k=50) + "\n\n")
            f.write("[Transformer]\n" + generate_from_neural(tf, vocab, p, device=device, top_k=50) + "\n\n")

    print(f"\n[DONE] Saved results to: {outdir}")
    print(f"[DONE] Saved generations to: {gen_path}")


if __name__ == "__main__":
    main()
