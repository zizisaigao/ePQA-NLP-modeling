#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part II-C helper: load GloVe (or any word->vector text file) and build a FIXED embedding
matrix aligned to Part I vocab, then inject it into your LM (RNN/LSTM/Transformer).

Expected vocab.json format (from Part I):
  {"itos": ["<pad>", "<unk>", "<bos>", "<eos>", ...]}

Supported vector formats:
  - GloVe .txt: each line "word v1 v2 ... vd" (NO header)
  - fastText .vec: optional header "N D" on first line (auto-detected)
"""

import argparse
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def load_itos(vocab_json: str) -> List[str]:
    with open(vocab_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "itos" not in obj:
        raise ValueError("vocab_json must contain key 'itos'")
    return obj["itos"]


def read_text_vectors(
    path: str,
    expect_dim: Optional[int] = None,
    max_vocab: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Read word vectors from a text file.
    Supports:
      - GloVe: no header
      - fastText .vec: first line 'n d'
    Returns: (vectors, dim)
    """
    vecs: Dict[str, np.ndarray] = {}
    dim: Optional[int] = None

    with io.open(path, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
        first = f.readline()
        if not first:
            raise ValueError("Empty vector file")
        parts = first.strip().split()

        # Detect header: two integers
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            dim = int(parts[1])
        else:
            if len(parts) <= 2:
                raise ValueError("First vector line is malformed")
            word = parts[0]
            vals = np.asarray(parts[1:], dtype=np.float32)
            dim = int(vals.shape[0])
            vecs[word] = vals

        if expect_dim is not None and dim != expect_dim:
            raise ValueError(f"Embedding dim mismatch: file dim={dim}, expect_dim={expect_dim}")

        for line in f:
            if max_vocab is not None and len(vecs) >= max_vocab:
                break
            ps = line.rstrip().split()
            if len(ps) <= 2:
                continue
            w = ps[0]
            v = np.asarray(ps[1:], dtype=np.float32)
            if v.shape[0] != dim:
                continue
            vecs[w] = v

    return vecs, dim


def build_glove_weight(
    itos: List[str],
    glove_path: str,
    emb_dim: int,
    seed: int = 13,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
    oov_strategy: str = "unk",   # "unk" or "random"
    max_glove_vocab: Optional[int] = None,
) -> Tuple[torch.FloatTensor, float]:
    """
    Build embedding weight aligned to itos. Returns (weight, coverage).

    oov_strategy:
      - "unk": map OOV words to unk vector (if exists), else to mean vector
      - "random": keep random init for OOV
    """
    rng = np.random.default_rng(seed)
    glove, dim = read_text_vectors(glove_path, expect_dim=emb_dim, max_vocab=max_glove_vocab)

    W = rng.normal(loc=0.0, scale=0.02, size=(len(itos), emb_dim)).astype(np.float32)

    # pad -> zeros
    if pad_token in itos:
        W[itos.index(pad_token)] = 0.0

    # unk fallback vector
    unk_vec = glove.get(unk_token, None)
    if unk_vec is None and len(glove) > 0:
        # Use mean vector as fallback (common practical choice)
        sample = list(glove.values())[:50000]
        unk_vec = np.mean(np.stack(sample, axis=0), axis=0).astype(np.float32)

    hit = 0
    for i, tok in enumerate(itos):
        v = glove.get(tok, None)
        if v is not None:
            W[i] = v
            hit += 1
        else:
            if oov_strategy == "unk" and tok != pad_token and unk_vec is not None:
                W[i] = unk_vec
            # else keep random

    coverage = hit / max(1, len(itos))
    return torch.tensor(W, dtype=torch.float32), coverage


def inject_fixed_embedding(model: nn.Module, weight: torch.FloatTensor) -> None:
    """
    Replace model's token embedding with fixed pretrained embedding.

    Compatible with your Part I naming:
      - RNNLM / LSTMLM: model.emb
      - TransformerLM: model.tok_emb
    """
    emb = None
    if hasattr(model, "emb") and isinstance(getattr(model, "emb"), nn.Embedding):
        emb = getattr(model, "emb")
        name = "emb"
    elif hasattr(model, "tok_emb") and isinstance(getattr(model, "tok_emb"), nn.Embedding):
        emb = getattr(model, "tok_emb")
        name = "tok_emb"
    else:
        raise ValueError("Cannot find embedding layer: expected model.emb or model.tok_emb")

    if tuple(emb.weight.shape) != tuple(weight.shape):
        raise ValueError(f"Shape mismatch: model {name} weight {tuple(emb.weight.shape)} vs provided {tuple(weight.shape)}")

    with torch.no_grad():
        emb.weight.copy_(weight.to(emb.weight.device))
    emb.weight.requires_grad = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_json", default="result_4_300d/vocab.json", help="Part I vocab.json (contains itos)")
    ap.add_argument("--glove_txt", required=True, help="GloVe (or other) text vectors file")
    ap.add_argument("--emb_dim", type=int, required=True, help="Embedding dimension (must match glove dim)")
    ap.add_argument("--out_pt", default="glove_aligned.pt", help="Output torch weight file")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--oov_strategy", choices=["unk", "random"], default="unk")
    ap.add_argument("--max_glove_vocab", type=int, default=0, help="Optional limit for reading glove (0=all)")
    args = ap.parse_args()

    itos = load_itos(args.vocab_json)
    max_vocab = None if args.max_glove_vocab <= 0 else args.max_glove_vocab
    W, cov = build_glove_weight(
        itos=itos,
        glove_path=args.glove_txt,
        emb_dim=args.emb_dim,
        seed=args.seed,
        oov_strategy=args.oov_strategy,
        max_glove_vocab=max_vocab,
    )
    out = Path(args.out_pt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": W, "coverage": cov, "emb_dim": args.emb_dim}, out)
    print(f"[saved] {out} (shape={tuple(W.shape)} coverage={cov:.2%})")


if __name__ == "__main__":
    main()
