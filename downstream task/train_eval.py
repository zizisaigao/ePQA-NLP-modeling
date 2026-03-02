from typing import Dict, Tuple, Optional, List
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def confusion_matrix_3(y_true: List[int], y_pred: List[int]) -> List[List[int]]:
    cm = [[0,0,0],[0,0,0],[0,0,0]]
    for t,p in zip(y_true, y_pred):
        if 0 <= t <= 2 and 0 <= p <= 2:
            cm[t][p] += 1
    return cm

def macro_f1_3(cm: List[List[int]]) -> float:
    f1s = []
    for c in range(3):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(3) if r != c)
        fn = sum(cm[c][r] for r in range(3) if r != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / 3.0

@torch.no_grad()
def evaluate_3class(model, dataloader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    y_true, y_pred = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids, attn)
        preds = out.pred_3class
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

    cm = confusion_matrix_3(y_true, y_pred)
    acc = sum(cm[i][i] for i in range(3)) / max(1, sum(sum(r) for r in cm))
    mf1 = macro_f1_3(cm)
    return {"acc": acc, "macro_f1": mf1, "confusion": cm}

def make_binary_targets(labels: torch.Tensor, stage: str) -> torch.Tensor:
    if stage == "stage0":
        # 0 vs {1,2}: target 0 if label==0 else 1
        return (labels != 0).long()
    if stage == "stage12":
        # 1 vs 2: only defined for labels in {1,2}; map 1->0, 2->1
        return (labels == 2).long()
    raise ValueError(stage)

def train_one_epoch_binary(
    model,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    stage: str,
    grad_clip: Optional[float] = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels3 = batch["labels"].to(device)

        if stage == "stage12":
            # filter to labels in {1,2}
            mask = (labels3 != 0)
            if mask.sum().item() == 0:
                continue
            input_ids = input_ids[mask]
            attn = attn[mask]
            labels3 = labels3[mask]

        targets = make_binary_targets(labels3, stage)

        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids, attn)
        logits = out.logits_stage0 if stage == "stage0" else out.logits_stage12
        loss = loss_fn(logits, targets)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(1, total_n)
