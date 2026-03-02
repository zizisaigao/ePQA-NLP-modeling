#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_vectors_to_vocab.py

Stream-align a large text embedding file (GloVe .txt / fastText .vec) to your Part I vocab order,
and save a torch .pt weight file that can be loaded quickly during Part II runs.

This avoids loading the full embedding file into memory (important for wiki-news-300d-1M.vec).

Input embedding formats supported:
  - GloVe: each line "word v1 v2 ... vD" (no header)
  - fastText .vec: optional header "N D" on first line

Output:
  - .pt containing: {"weight": FloatTensor [|vocab|, D], "coverage": float, "emb_dim": int, "source": str}
Optional:
  - write a filtered .vec/.txt file containing only tokens in vocab (for inspection)

Example:
  python align_vectors_to_vocab.py --vocab_json ../processed_data/corpus.json \
    --vec_path glove.2024.wikigiga.300d.txt --emb_dim 300 --out_pt aligned_public_glove_300.pt

  python align_vectors_to_vocab.py --vocab_json ../processed_data/corpus.json \
    --vec_path wiki-news-300d-1M.vec --emb_dim 300 --out_pt aligned_public_ft_300.pt
"""

import argparse
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def load_itos(vocab_json: str) -> List[str]:
    with open(vocab_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "itos" not in obj:
        raise ValueError("vocab_json must contain key 'itos'")
    return obj["itos"]


def stream_align_to_vocab(
    itos: List[str],
    vec_path: str,
    emb_dim: int,
    seed: int = 1234,
    pad_token: str = "<pad>",
) -> Tuple[torch.FloatTensor, float, int]:
    """
    Stream read vec_path and fill weight rows for tokens in vocab.
    OOV rows stay random normal; pad row set to zeros if present.

    Returns: (weight, coverage, hits)
    """
    stoi = {t: i for i, t in enumerate(itos)}
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.02, size=(len(itos), emb_dim)).astype(np.float32)

    if pad_token in stoi:
        W[stoi[pad_token]] = 0.0

    hit = 0
    filled = np.zeros((len(itos),), dtype=bool)

    def process_line(line: str) -> None:
        nonlocal hit
        parts = line.rstrip().split()
        if len(parts) <= emb_dim:
            return
        tok = parts[0]
        idx = stoi.get(tok)
        if idx is None or filled[idx]:
            return
        try:
            vec = np.asarray(parts[1:], dtype=np.float32)
        except ValueError:
            return
        if vec.shape[0] != emb_dim:
            return
        W[idx] = vec
        filled[idx] = True
        hit += 1

    with io.open(vec_path, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
        first = f.readline()
        if not first:
            raise ValueError("Empty embedding file")

        # Detect fastText header: "N D"
        ps = first.strip().split()
        if not (len(ps) == 2 and ps[0].isdigit() and ps[1].isdigit()):
            process_line(first)

        for line in f:
            process_line(line)
            # Early stop if we've filled almost everything (optional; keep simple)
            # if hit >= len(itos):
            #     break

    coverage = hit / max(1, len(itos))
    return torch.tensor(W, dtype=torch.float32), float(coverage), int(hit)


def write_filtered_vectors(
    itos: List[str],
    vec_path: str,
    emb_dim: int,
    out_vec: str,
) -> float:
    """
    Write a filtered embedding text file containing only tokens in vocab.
    Returns coverage.
    """
    stoi = {t: i for i, t in enumerate(itos)}
    seen = set()
    hit = 0

    outp = Path(out_vec)
    outp.parent.mkdir(parents=True, exist_ok=True)

    def maybe_write(line: str, fout) -> None:
        nonlocal hit
        parts = line.rstrip().split()
        if len(parts) <= emb_dim:
            return
        tok = parts[0]
        if tok in stoi and tok not in seen:
            # validate dim
            if len(parts) - 1 != emb_dim:
                return
            fout.write(line if line.endswith("\n") else (line + "\n"))
            seen.add(tok)
            hit += 1

    with io.open(vec_path, "r", encoding="utf-8", newline="\n", errors="ignore") as fin, \
         io.open(outp, "w", encoding="utf-8", newline="\n") as fout:
        first = fin.readline()
        if not first:
            raise ValueError("Empty embedding file")

        ps = first.strip().split()
        if len(ps) == 2 and ps[0].isdigit() and ps[1].isdigit():
            # keep header out (optional). Comment out next line if you want header.
            # fout.write(first)
            pass
        else:
            maybe_write(first, fout)

        for line in fin:
            maybe_write(line, fout)

    coverage = hit / max(1, len(itos))
    return float(coverage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_json", default="result_4_300d/vocab.json", help="Part I vocab json (contains itos)")
    ap.add_argument("--vec_path", default="wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt", help="Embedding file (.txt or .vec)")
    ap.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension (e.g., 300)")
    ap.add_argument("--out_pt", default="wiki_giga_aligned_embeddings.pt", help="Output .pt path for aligned weight")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--pad_token", default="<pad>")
    ap.add_argument("--out_vec", default="", help="Optional: output filtered embedding file (only vocab tokens)")

    args = ap.parse_args()

    itos = load_itos(args.vocab_json)

    weight, cov, hit = stream_align_to_vocab(
        itos=itos,
        vec_path=args.vec_path,
        emb_dim=args.emb_dim,
        seed=args.seed,
        pad_token=args.pad_token,
    )

    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"weight": weight, "coverage": cov, "emb_dim": args.emb_dim, "source": str(args.vec_path)},
        out_pt,
    )
    print(f"[saved] {out_pt} shape={tuple(weight.shape)} coverage={cov:.2%} hits={hit}")

    if args.out_vec:
        cov2 = write_filtered_vectors(itos, args.vec_path, args.emb_dim, args.out_vec)
        print(f"[saved] {args.out_vec} coverage={cov2:.2%}")


if __name__ == "__main__":
    main()
    #[saved] wiki_giga_aligned_embeddings.pt shape=(30000, 300) coverage=44.95% hits=13484