"""Common neural-language-model utilities for ST5230 Assignment 1 Part II.

This module is extracted from Part I code and intentionally excludes the n-gram model.
It provides:
- data helpers (tokenize, load_fixed_splits, batchify)
- Vocab class (loading/encoding)
- neural LMs (RNN/LSTM/Transformer)
- training + perplexity evaluation utilities
"""

import os
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

RE_MULTI_SPACE = re.compile(r"\s+")


def set_seed(seed: int = 1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def ensure_text_clean_from_qa(
    df: pd.DataFrame,
    include_context: bool = True,
    refuse_text: str = "No answer based on the provided candidate.",
    # partial_text: str = "The candidate provides helpful information but does not fully answer the question."
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

class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad: int
    unk: int
    bos: int
    eos: int

    def __init__(self, stoi: Dict[str, int], itos: List[str], pad: int, unk: int, bos: int, eos: int):
        self.stoi = stoi
        self.itos = itos
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        
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
'''
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
'''
#part3用的transformer
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 4,
                 dim_ff: int = 1024, dropout: float = 0.2, max_len: int = 2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=False
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.d_model = d_model

    def encode(self, input_ids: torch.Tensor, attention_mask=None):
        """
        input_ids: [B,T] (Part3)  -> transformer expects [T,B]
        returns:   [B,T,D]
        """
        if input_ids.dim() != 2:
            raise ValueError(f"encode expects [B,T], got {input_ids.shape}")

        x = input_ids.transpose(0, 1)  # [T,B]
        T, B = x.size()

        pos = torch.arange(0, T, device=x.device).unsqueeze(1).expand(T, B)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.dropout(h)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        out = self.tr(h, mask=mask)    # [T,B,D]

        return out.transpose(0, 1)     # [B,T,D]


    def forward(self, x: torch.Tensor):
        # 支持两种：训练LM时你可能传 [T,B]；Part3 传 [B,T]
        if x.dim() != 2:
            raise ValueError(f"forward expects 2D tensor, got {x.shape}")

        # 若是 [T,B]，先转成 [B,T] 复用 encode
        if x.size(0) > x.size(1):   # 训练LM常见 T > B
            x_bt = x.transpose(0, 1)
        else:
            x_bt = x

        hidden_bt = self.encode(x_bt)          # [B,T,D]
        logits_tb = self.fc(hidden_bt.transpose(0, 1))  # [T,B,V]
        return logits_tb
    
    
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
