import argparse
import json
import os
from typing import Dict, Any, Optional

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import Vocab, QACLabelDataset, collate_batch
from lm_loader import load_lm
from classifier import TwoStageClassifier
from train_eval import train_one_epoch_binary, evaluate_3class
from utils import set_seed, save_json, EarlyStopper

def infer_hidden_dim(model: nn.Module) -> int:
    # Try common attributes first
    for attr in ["d_model", "hidden_size", "emb_dim", "embed_dim", "model_dim"]:
        if hasattr(model, attr):
            v = getattr(model, attr)
            if isinstance(v, int):
                return v
    # Fallback: look for an embedding weight shape
    for name, p in model.named_parameters():
        if "embed" in name and p.dim() == 2:
            return int(p.size(1))
    raise ValueError("Cannot infer hidden dim from LM. Provide --hidden_dim explicitly.")

def set_freeze_mode(model: nn.Module, mode: str, unfreeze_last_k: int = 0) -> None:
    # mode: freeze | finetune_all | finetune_lastk
    for p in model.parameters():
        p.requires_grad = False

    if mode == "freeze":
        return

    if mode == "finetune_all":
        for p in model.parameters():
            p.requires_grad = True
        return

    if mode == "finetune_lastk":
        # Try to unfreeze last K transformer blocks if accessible
        # Common patterns: model.blocks / model.layers / model.encoder.layers
        candidates = []
        if hasattr(model, "blocks"):
            candidates = list(getattr(model, "blocks"))
        elif hasattr(model, "layers"):
            candidates = list(getattr(model, "layers"))
        elif hasattr(model, "encoder") and hasattr(getattr(model, "encoder"), "layers"):
            candidates = list(getattr(getattr(model, "encoder"), "layers"))
        elif hasattr(model, "tr") and hasattr(getattr(model, "tr"), "layers"):
            candidates = list(getattr(getattr(model, "tr"), "layers"))
        if candidates:
            for blk in candidates[-unfreeze_last_k:]:
                for p in blk.parameters():
                    p.requires_grad = True
        else:
            # Fallback: just unfreeze everything (but warn)
            print("[run_part3] Warning: could not locate transformer blocks; falling back to finetune_all.")
            for p in model.parameters():
                p.requires_grad = True
        return

    raise ValueError(f"Unknown mode: {mode}")

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", default="data", help="directory containing train.csv/dev.csv/test.csv")
    ap.add_argument("--vocab_json", default="result_4_300d/vocab.json", help="Part1/2 vocab json")
    ap.add_argument("--lm_pt", default="part3/best_transformer_public_finetune.pt", help="path to best LM checkpoint (.pt)")

    ap.add_argument("--model_py", default="part2/common_part1_imports.py", help="path to python file defining your Transformer class (if needed)")
    ap.add_argument("--model_class", default=None, help="class name of the Transformer model (if needed)")
    ap.add_argument("--model_kwargs_json", default=None, help="json string or path to json file with model init kwargs")

    ap.add_argument("--out_dir", default="part3_results", help="where to write metrics + checkpoints")

    ap.add_argument("--mode", choices=["freeze", "finetune_lastk", "finetune_all"], default="finetune_all")
    ap.add_argument("--unfreeze_last_k", type=int, default=2)

    ap.add_argument("--pool", choices=["mean", "eos"], default="mean")
    ap.add_argument("--hidden_dim", type=int, default=0, help="set if cannot infer from model")

    ap.add_argument("--include_title", action="store_true")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--class_weight_stage0", action="store_true", default=True, help="use class-weighted CE for stage0")
    ap.add_argument("--class_weight_stage12", action="store_true", default=True, help="use class-weighted CE for stage12")

    ap.add_argument("--patience", type=int, default=2)

    return ap.parse_args()

def load_kwargs(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    # If it's a file path
    if os.path.exists(s):
        with open(s, "r", encoding="utf-8") as f:
            return json.load(f)
    # Else treat as JSON string
    return json.loads(s)

def compute_binary_class_weights(dset, stage: str) -> torch.Tensor:
    # Returns tensor([w0, w1]) where higher weight for rarer class.
    import numpy as np
    labels = dset.df["label"].astype(int).to_numpy()
    if stage == "stage0":
        y = (labels != 0).astype(int)
    else:
        y = (labels == 2).astype(int)
        y = y[labels != 0]
    counts = np.bincount(y, minlength=2).astype(float)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float)

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    vocab = Vocab.from_json(args.vocab_json)

    train_csv = os.path.join(args.data_dir, "train.csv")
    dev_csv = os.path.join(args.data_dir, "dev.csv")
    test_csv = os.path.join(args.data_dir, "test.csv")

    train_ds = QACLabelDataset(train_csv, vocab, max_len=args.max_len, include_title=args.include_title)
    dev_ds   = QACLabelDataset(dev_csv, vocab, max_len=args.max_len, include_title=args.include_title)
    test_ds  = QACLabelDataset(test_csv, vocab, max_len=args.max_len, include_title=args.include_title)

    collate = lambda b: collate_batch(b, pad_id=vocab.pad)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    dev_dl   = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    # 读取checkpoint里保存的构造信息（model_class/model_kwargs）
    ckpt_preview = torch.load(args.lm_pt, map_location="cpu")

    model_kwargs = load_kwargs(args.model_kwargs_json)  # 允许你手动传入覆盖
    if isinstance(ckpt_preview, dict):
        if args.model_class is None and "model_class" in ckpt_preview:
            args.model_class = ckpt_preview["model_class"]  # 这里会是 TransformerLM
        if (not model_kwargs) and ("model_kwargs" in ckpt_preview):
            model_kwargs = ckpt_preview["model_kwargs"]      # 这里就是你贴的那串dict
    lm = load_lm(args.lm_pt, device=device, model_py=args.model_py, model_class=args.model_class, model_kwargs=model_kwargs)

    # Freeze / finetune configuration
    set_freeze_mode(lm, args.mode, unfreeze_last_k=args.unfreeze_last_k)

    hidden_dim = args.hidden_dim if args.hidden_dim > 0 else infer_hidden_dim(lm)
    model = TwoStageClassifier(lm=lm, hidden_dim=hidden_dim, pool=args.pool, eos_id=vocab.eos).to(device)

    # Optimizer: only update trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # Loss functions (binary CE over logits with optional class weights)
    w0 = compute_binary_class_weights(train_ds, "stage0") if args.class_weight_stage0 else None
    w12 = compute_binary_class_weights(train_ds, "stage12") if args.class_weight_stage12 else None
    loss0 = nn.CrossEntropyLoss(weight=(w0.to(device) if w0 is not None else None))
    loss12 = nn.CrossEntropyLoss(weight=(w12.to(device) if w12 is not None else None))

    stopper = EarlyStopper(patience=args.patience, mode="max")

    history = []
    total_train_s = 0.0
    total_dev_s = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss0 = train_one_epoch_binary(model, train_dl, device, optimizer, loss0, stage="stage0", grad_clip=args.grad_clip)
        train_loss12 = train_one_epoch_binary(model, train_dl, device, optimizer, loss12, stage="stage12", grad_clip=args.grad_clip)
        train_s = time.time() - t0
        total_train_s += train_s

        t1 = time.time()
        dev_metrics = evaluate_3class(model, dev_dl, device)
        dev_s = time.time() - t1
        total_dev_s += dev_s
        row = {
            "epoch": epoch,
            "train_loss_stage0": train_loss0,
            "train_loss_stage12": train_loss12,
            "dev_acc": dev_metrics["acc"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "train_s": train_s,
            "dev_s": dev_s,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        # early stop on dev_macro_f1
        stop = stopper.step(float(dev_metrics["macro_f1"]), state={"model_state": model.state_dict()})
        if stop:
            print(f"[run_part3] Early stopping at epoch {epoch}. Best dev_macro_f1={stopper.best:.4f}")
            break

    # restore best
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state["model_state"], strict=True)

    dev_final = evaluate_3class(model, dev_dl, device)
    test_final = evaluate_3class(model, test_dl, device)

    t2 = time.time()
    dev_final = evaluate_3class(model, dev_dl, device)
    dev_final_s = time.time() - t2

    t3 = time.time()
    test_final = evaluate_3class(model, test_dl, device)
    test_final_s = time.time() - t3

    out = {
        "args": vars(args),
        "best_dev_macro_f1": stopper.best,
        "dev": dev_final,
        "test": test_final,
        "dev_s": dev_final_s,
        "test_s": test_final_s,
        "total_train_s": total_train_s,
        "total_dev_s": total_dev_s,
        "history": history,
    }

    tag = f"{args.mode}{('_k'+str(args.unfreeze_last_k)) if args.mode=='finetune_lastk' else ''}_{args.pool}"
    results_path = os.path.join(args.out_dir, f"results_part3_{tag}.json")
    ckpt_path    = os.path.join(args.out_dir, f"best_two_stage_classifier_{tag}.pt")
    save_json(out, results_path)
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)
    print("[run_part3] Saved:", results_path)

    #print("[run_part3] Saved:", os.path.join(args.out_dir, "results_part3.json"))

if __name__ == "__main__":
    main()
