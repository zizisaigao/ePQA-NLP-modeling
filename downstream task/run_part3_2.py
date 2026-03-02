"""
Part 3 runner (compare multiple LM checkpoints + two modes: freeze vs finetune_all)

Usage (Windows PowerShell):
python -m part3.part3.2.py `
  --data_dir ..\data `
  --vocab_json ..\result_4_300d\vocab.json `
  --lm_pts "..\result_4_300d\best_transformer.pt,..\result_4_300d\best_lstm.pt" `
  --model_py ..\part2\models.py `
  --out_dir part3_results_compare `
  --modes freeze,finetune_all `
  --pool mean `
  --hidden_dim 300
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow "python part3/run_part3.py"
if __package__ is None or __package__ == "":
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _proj_root = os.path.dirname(_this_dir)
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)

from part3.data import Vocab, QACLabelDataset, collate_batch
from part3.lm_loader import load_lm
from part3.classifier import TwoStageClassifier
from part3.train_eval import train_one_epoch_binary, evaluate_3class
from part3.utils import set_seed, save_json, EarlyStopper


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", required=True, help="directory containing train.csv/dev.csv/test.csv")
    ap.add_argument("--vocab_json", required=True, help="Part1/2 vocab json")

    # IMPORTANT: allow multiple checkpoints
    ap.add_argument("--lm_pts", required=True,
                    help="Comma-separated list of LM checkpoint paths (.pt). Example: a.pt,b.pt,c.pt")

    # needed if checkpoint stores only state_dict (common). If checkpoint stores model object, can be omitted.
    ap.add_argument("--model_py", default=None,
                    help="Path to python file defining model classes (TransformerLM / RNNLM / LSTMLM), if needed")

    ap.add_argument("--out_dir", default="part3_results_compare", help="root output directory")
    ap.add_argument("--modes", default="freeze,finetune_all",
                    help="Comma-separated modes to run: freeze,finetune_all")
    ap.add_argument("--pool", choices=["mean", "eos"], default="mean")

    # downstream / training
    ap.add_argument("--include_title", action="store_true")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--patience", type=int, default=3)

    # if hidden dim cannot be inferred reliably, force it
    ap.add_argument("--hidden_dim", type=int, default=0, help="Representation dim D. Recommend set to 300 for your setup.")

    # Optional: class-weighting for imbalance
    ap.add_argument("--class_weight_stage0", action="store_true")
    ap.add_argument("--class_weight_stage12", action="store_true")

    return ap.parse_args()


def infer_hidden_dim(model: nn.Module) -> int:
    for attr in ["d_model", "hidden_size", "hid_dim", "emb_dim", "embed_dim", "model_dim"]:
        if hasattr(model, attr) and isinstance(getattr(model, attr), int):
            return int(getattr(model, attr))
    # Try embedding weight
    for name, p in model.named_parameters():
        if ("tok_emb" in name or "embed" in name or "emb" in name) and p.dim() == 2:
            return int(p.size(1))
    raise ValueError("Cannot infer hidden dim from LM. Please pass --hidden_dim explicitly.")


def set_freeze_mode(lm: nn.Module, mode: str) -> None:
    if mode == "freeze":
        for p in lm.parameters():
            p.requires_grad = False
        return
    if mode == "finetune_all":
        for p in lm.parameters():
            p.requires_grad = True
        return
    raise ValueError(f"Unsupported mode: {mode}. Use freeze or finetune_all.")


def compute_binary_class_weights(train_ds: QACLabelDataset, stage: str) -> torch.Tensor:
    import numpy as np
    labels = train_ds.df["label"].astype(int).to_numpy()
    if stage == "stage0":
        y = (labels != 0).astype(int)
    else:
        y = (labels == 2).astype(int)
        y = y[labels != 0]
    counts = np.bincount(y, minlength=2).astype(float)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float)


def sanitize_name(path: str) -> str:
    base = os.path.basename(path)
    base = base.replace(".pt", "")
    return base


def run_one_setting(
    lm_pt: str,
    mode: str,
    args,
    device: torch.device,
    vocab: Vocab,
    train_dl: DataLoader,
    dev_dl: DataLoader,
    test_dl: DataLoader,
) -> Dict[str, Any]:
    # load ckpt preview for model_class/model_kwargs
    ckpt = torch.load(lm_pt, map_location="cpu", weights_only=True)
    
    model_class = None
    model_kwargs = {}

    if isinstance(ckpt, dict):
        model_class = ckpt.get("model_class", None)
        model_kwargs = ckpt.get("model_kwargs", {}) or {}
    
    # 如果 part1 ckpt 没有 model_class/model_kwargs，就用文件名猜 + 用你记录的hparams补齐
    if model_class is None:
        name = os.path.basename(lm_pt).lower()
        if "transformer" in name:
            model_class = "TransformerLM"
            model_kwargs = {"vocab_size": 30000, "d_model": 300, "nhead": 4, "num_layers": 4,
                            "dim_ff": 1024, "dropout": 0.2, "max_len": 2048}
        elif "lstm" in name:
            model_class = "LSTMLM"
            model_kwargs = {"vocab_size": 30000, "emb_dim": 300, "hid_dim": 300, "num_layers": 1, "dropout": 0.2}
        elif "rnn" in name:
            model_class = "RNNLM"
            model_kwargs = {"vocab_size": 30000, "emb_dim": 300, "hid_dim": 300, "num_layers": 1, "dropout": 0.2}
        else:
            raise ValueError(f"Cannot infer model_class from filename: {lm_pt}. Please pass --model_class/--model_kwargs_json.")
    # allow ckpt to point to its vocab_json; you still control args.vocab_json
    # but we record it for traceability
    # load LM
    lm = load_lm(
        ckpt_path=lm_pt,
        device=device,
        model_py=args.model_py,
        model_class=model_class,
        model_kwargs=model_kwargs,
    )

    set_freeze_mode(lm, mode)

    hidden_dim = args.hidden_dim if args.hidden_dim > 0 else infer_hidden_dim(lm)
    model = TwoStageClassifier(lm=lm, hidden_dim=hidden_dim, pool=args.pool, eos_id=vocab.eos).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    w0 = compute_binary_class_weights(train_ds=train_dl.dataset, stage="stage0") if args.class_weight_stage0 else None
    w12 = compute_binary_class_weights(train_ds=train_dl.dataset, stage="stage12") if args.class_weight_stage12 else None
    loss0 = nn.CrossEntropyLoss(weight=(w0.to(device) if w0 is not None else None))
    loss12 = nn.CrossEntropyLoss(weight=(w12.to(device) if w12 is not None else None))

    stopper = EarlyStopper(patience=args.patience, mode="max")
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss0 = train_one_epoch_binary(
            model, train_dl, device, optimizer, loss0, stage="stage0", grad_clip=args.grad_clip
        )
        train_loss12 = train_one_epoch_binary(
            model, train_dl, device, optimizer, loss12, stage="stage12", grad_clip=args.grad_clip
        )

        dev_metrics = evaluate_3class(model, dev_dl, device)
        row = {
            "epoch": epoch,
            "train_loss_stage0": float(train_loss0),
            "train_loss_stage12": float(train_loss12),
            "dev_acc": float(dev_metrics["acc"]),
            "dev_macro_f1": float(dev_metrics["macro_f1"]),
        }
        history.append(row)
        print(json.dumps({"lm": os.path.basename(lm_pt), "mode": mode, **row}, ensure_ascii=False))

        stop = stopper.step(float(dev_metrics["macro_f1"]), state={"model_state": model.state_dict()})
        if stop:
            break

    train_time = time.time() - t0

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state["model_state"], strict=True)

    dev_final = evaluate_3class(model, dev_dl, device)
    test_final = evaluate_3class(model, test_dl, device)

    out = {
        "lm_pt": lm_pt,
        "mode": mode,
        "pool": args.pool,
        "hidden_dim": hidden_dim,
        "train_time_s": train_time,
        "best_dev_macro_f1": stopper.best,
        "dev": dev_final,
        "test": test_final,
        "history": history,
    }

    return out


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    # data/vocab
    vocab = Vocab.from_json(args.vocab_json)

    train_csv = os.path.join(args.data_dir, "train.csv")
    dev_csv = os.path.join(args.data_dir, "dev.csv")
    test_csv = os.path.join(args.data_dir, "test.csv")

    train_ds = QACLabelDataset(train_csv, vocab, max_len=args.max_len, include_title=args.include_title)
    dev_ds = QACLabelDataset(dev_csv, vocab, max_len=args.max_len, include_title=args.include_title)
    test_ds = QACLabelDataset(test_csv, vocab, max_len=args.max_len, include_title=args.include_title)

    collate = lambda b: collate_batch(b, pad_id=vocab.pad)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    lm_pts = [p.strip() for p in args.lm_pts.split(",") if p.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    summary_rows = []

    for lm_pt in lm_pts:
        lm_name = sanitize_name(lm_pt)
        for mode in modes:
            run_dir = os.path.join(args.out_dir, lm_name, mode)
            os.makedirs(run_dir, exist_ok=True)

            result = run_one_setting(
                lm_pt=lm_pt,
                mode=mode,
                args=args,
                device=device,
                vocab=vocab,
                train_dl=train_dl,
                dev_dl=dev_dl,
                test_dl=test_dl,
            )
            save_json(result, os.path.join(run_dir, "results_part3.json"))

            summary_rows.append({
                "lm": lm_name,
                "mode": mode,
                "pool": args.pool,
                "dev_acc": float(result["dev"]["acc"]),
                "dev_macro_f1": float(result["dev"]["macro_f1"]),
                "test_acc": float(result["test"]["acc"]),
                "test_macro_f1": float(result["test"]["macro_f1"]),
                "train_time_s": float(result["train_time_s"]),
            })

    save_json({"summary": summary_rows}, os.path.join(args.out_dir, "summary.json"))
    print("[run_part3] Wrote summary:", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()