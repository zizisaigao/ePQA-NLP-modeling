# part2/embedding_utils.py
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

def load_vocab_json(vocab_path: str) -> List[str]:
    """Load itos from Part I saved vocab.json: {"itos": [...]}"""
    with open(vocab_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["itos"]

def load_text_vectors(path: str, max_vocab: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Load text format vectors: each line: token val1 val2 ...
    Supports GloVe/fastText-like plain text.
    """
    vecs = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) <= 2:
                continue
            w = parts[0]
            arr = np.asarray(parts[1:], dtype=np.float32)
            vecs[w] = arr
            if max_vocab and len(vecs) >= max_vocab:
                break
    return vecs

def build_embedding_matrix(
    itos: List[str],
    pretrained: Dict[str, np.ndarray],
    dim: int,
    seed: int = 1234,
    unk_token: str = "<unk>",
) -> np.ndarray:
    """
    Align pretrained vectors to your vocab order (itos).
    OOV: random normal (or you can copy unk vector if exists).
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.02, size=(len(itos), dim)).astype(np.float32)

    # If pretrained has unk, use it as fallback prototype
    unk_vec = pretrained.get(unk_token, None)

    hit = 0
    for i, tok in enumerate(itos):
        v = pretrained.get(tok)
        if v is not None and v.shape[0] == dim:
            W[i] = v
            hit += 1
        elif unk_vec is not None and unk_vec.shape[0] == dim:
            # optional: map OOV to unk vector
            # W[i] = unk_vec
            pass

    coverage = hit / max(1, len(itos))
    return W, coverage

def _get_embedding_module(model: nn.Module) -> nn.Embedding:
    """
    Match your Part I model definitions:
      RNNLM.emb, LSTMLM.emb, TransformerLM.tok_emb
    """
    if hasattr(model, "emb") and isinstance(model.emb, nn.Embedding):
        return model.emb
    if hasattr(model, "tok_emb") and isinstance(model.tok_emb, nn.Embedding):
        return model.tok_emb
    raise ValueError("Cannot find embedding layer (expected .emb or .tok_emb).")

def inject_pretrained_embeddings(
    model: nn.Module,
    emb_matrix: np.ndarray,
    freeze: bool,
) -> None:
    emb = _get_embedding_module(model)
    if emb.weight.shape != torch.Size(emb_matrix.shape):
        raise ValueError(f"Shape mismatch: model emb {tuple(emb.weight.shape)} vs matrix {emb_matrix.shape}")
    with torch.no_grad():
        emb.weight.copy_(torch.from_numpy(emb_matrix))
    emb.weight.requires_grad = (not freeze)