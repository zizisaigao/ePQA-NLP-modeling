import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

_WS_RE = re.compile(r"\s+")

def simple_tokenize(text: str) -> List[str]:
    # Keep this minimal to avoid mismatch with your Part1/2 pipeline.
    # Assumes your vocab was built from similar whitespace tokenization.
    text = text.strip()
    text = _WS_RE.sub(" ", text)
    return text.split(" ") if text else []

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad: int
    unk: int
    bos: int
    eos: int

    @classmethod
    def from_json(cls, path: str) -> "Vocab":
        import json
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # Accept a few common formats
        if "stoi" in obj and "itos" in obj:
            stoi = obj["stoi"]
            itos = obj["itos"]
        elif "token_to_id" in obj and "id_to_token" in obj:
            stoi = obj["token_to_id"]
            itos = obj["id_to_token"]
        elif "itos" in obj:
            itos = obj["itos"]
            stoi = {t: i for i, t in enumerate(itos)}
        else:
            raise ValueError(f"Unrecognized vocab json format: keys={list(obj.keys())}")

        def get_id(tok: str) -> int:
            if tok in obj:            # some formats store ids directly
                return int(obj[tok])
            if tok in stoi:
                return int(stoi[tok])
            # fallback: search in itos
            try:
                return int(itos.index(tok))
            except ValueError:
                raise ValueError(f"Missing special token {tok} in vocab (and not provided as id).")

        pad = get_id("<pad>")
        unk = get_id("<unk>")
        bos = get_id("<bos>")
        eos = get_id("<eos>")
        return cls(stoi=stoi, itos=itos, pad=pad, unk=unk, bos=bos, eos=eos)

def encode_text(
    vocab: Vocab,
    text: str,
    max_len: int,
    add_bos_eos: bool = True
) -> Tuple[List[int], List[int]]:
    toks = simple_tokenize(text)
    ids = []
    if add_bos_eos:
        ids.append(vocab.bos)
    for t in toks:
        ids.append(vocab.stoi.get(t, vocab.unk))
    if add_bos_eos:
        ids.append(vocab.eos)

    if len(ids) > max_len:
        ids = ids[:max_len]
        # Ensure last token is eos if we truncated and eos exists
        if add_bos_eos:
            ids[-1] = vocab.eos

    attn = [1] * len(ids)
    return ids, attn

class QACLabelDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        vocab: Vocab,
        max_len: int = 256,
        include_title: bool = False,
        text_template: str = "question: {question} candidate: {candidate}",
    ):
        self.df = pd.read_csv(csv_path)
        required = {"question", "candidate", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns {missing}. Found {list(self.df.columns)}")

        self.vocab = vocab
        self.max_len = max_len
        self.include_title = include_title
        self.text_template = text_template

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fields = {
            "question": str(row["question"]),
            "candidate": str(row["candidate"]),
            "title": str(row["title"]) if "title" in self.df.columns else "",
        }
        if self.include_title:
            text = f"title: {fields['title']} " + self.text_template.format(**fields)
        else:
            text = self.text_template.format(**fields)

        ids, attn = encode_text(self.vocab, text, self.max_len, add_bos_eos=True)
        label = int(row["label"])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }

def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].numel() for x in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.stack([x["label"] for x in batch], dim=0)

    for i, x in enumerate(batch):
        L = x["input_ids"].numel()
        input_ids[i, :L] = x["input_ids"]
        attention_mask[i, :L] = x["attention_mask"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
