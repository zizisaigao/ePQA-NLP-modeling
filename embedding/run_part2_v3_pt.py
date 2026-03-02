# part2/run_part2_v3_pt.py
# Like run_part2_v2_fixed.py, but supports loading pre-aligned embedding weights from a .pt file
# produced by align_vectors_to_vocab.py (recommended for huge public embeddings).
#
# emb_mode:
#   - scratch_trainable: random init, trainable
#   - self_fixed: load embeddings (txt/vec/pt), freeze
#   - public_fixed: load embeddings (txt/vec/pt), freeze
#   - public_finetune: load embeddings (txt/vec/pt), trainable
#
# If emb_path ends with ".pt", expected keys: {"weight": FloatTensor, "coverage": float, "emb_dim": int}

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from embedding_utils import load_vocab_json, load_text_vectors, build_embedding_matrix, inject_pretrained_embeddings

from common_part1_imports import (
    set_seed,
    load_fixed_splits,
    tokenize,
    Vocab,
    flatten_token_lists,
    batchify,
    get_batch,
    eval_ppl_neural,
    RNNLM,
    LSTMLM,
    TransformerLM,
)


def _get_embedding_module(model: nn.Module) -> nn.Embedding:
    if hasattr(model, "emb") and isinstance(model.emb, nn.Embedding):
        return model.emb
    if hasattr(model, "tok_emb") and isinstance(model.tok_emb, nn.Embedding):
        return model.tok_emb
    raise ValueError("Cannot find embedding layer (expected .emb or .tok_emb).")


def inject_pretrained_weight_torch(model: nn.Module, weight: torch.Tensor, freeze: bool) -> None:
    emb = _get_embedding_module(model)
    if tuple(emb.weight.shape) != tuple(weight.shape):
        raise ValueError(f"Shape mismatch: model emb {tuple(emb.weight.shape)} vs weight {tuple(weight.shape)}")
    with torch.no_grad():
        emb.weight.copy_(weight.to(emb.weight.device))
    emb.weight.requires_grad = (not freeze)


def build_stream_from_df(df, vocab: Vocab):
    docs = [vocab.encode(tokenize(t) + ["<eos>"]) for t in df["text_clean"].astype(str).tolist()]
    stream = flatten_token_lists(docs, vocab.bos, vocab.eos)
    return stream


def _is_rnn_like(model: nn.Module) -> bool:
    return isinstance(model, (RNNLM, LSTMLM))


def train_neural_track(
    model: nn.Module,
    train_data: torch.Tensor,
    dev_data: torch.Tensor,
    seq_len: int,
    epochs: int,
    lr: float,
    clip: float,
    device: torch.device,
    log_every: int = 200,
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_dev = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None

    epoch_hist: List[Dict[str, Any]] = []
    grad_norm_hist: List[Dict[str, Any]] = []
    nan_or_inf = False

    model.to(device)

    for ep in range(1, epochs + 1):
        model.train()
        start = time.time()
        total_loss = 0.0
        total_tokens = 0
        steps = 0

        if _is_rnn_like(model):
            h = None
            iterator = tqdm(range(0, train_data.size(0) - 1, seq_len), desc=f"Epoch {ep}")
            for i in iterator:
                x, y = get_batch(train_data, i, seq_len)
                optimizer.zero_grad()
                logits, h = model(x, h)

                if isinstance(h, tuple):
                    h = tuple(t.detach() for t in h)
                else:
                    h = h.detach() if h is not None else None

                loss = criterion(logits.view(-1, logits.size(-1)), y.reshape(-1))
                finite = torch.isfinite(loss) if torch.is_tensor(loss) else math.isfinite(float(loss))
                if not bool(finite):
                    nan_or_inf = True
                    break

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
                steps += 1

                if log_every > 0 and steps % log_every == 0:
                    cur_nll = total_loss / max(1, total_tokens)
                    grad_norm_hist.append({"epoch": ep, "step": steps, "grad_norm": float(grad_norm)})
                    iterator.set_postfix({"train_ppl": f"{math.exp(cur_nll):.2f}", "grad": f"{float(grad_norm):.2f}"})
        else:
            iterator = tqdm(range(0, train_data.size(0) - 1, seq_len), desc=f"Epoch {ep}")
            for i in iterator:
                x, y = get_batch(train_data, i, seq_len)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.reshape(-1))

                finite = torch.isfinite(loss) if torch.is_tensor(loss) else math.isfinite(float(loss))
                if not bool(finite):
                    nan_or_inf = True
                    break

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
                steps += 1

                if log_every > 0 and steps % log_every == 0:
                    cur_nll = total_loss / max(1, total_tokens)
                    grad_norm_hist.append({"epoch": ep, "step": steps, "grad_norm": float(grad_norm)})
                    iterator.set_postfix({"train_ppl": f"{math.exp(cur_nll):.2f}", "grad": f"{float(grad_norm):.2f}"})

        if nan_or_inf:
            epoch_time = time.time() - start
            train_nll = total_loss / max(1, total_tokens)
            epoch_hist.append(
                {
                    "epoch": ep,
                    "train_nll": float(train_nll) if total_tokens > 0 else None,
                    "train_ppl": float(math.exp(train_nll)) if total_tokens > 0 else None,
                    "dev_ppl": None,
                    "epoch_time_s": float(epoch_time),
                    "note": "stopped: nan_or_inf",
                }
            )
            break

        train_nll = total_loss / max(1, total_tokens)
        train_ppl = math.exp(train_nll)
        dev_ppl = eval_ppl_neural(model, dev_data, seq_len, device)
        epoch_time = time.time() - start

        epoch_hist.append(
            {
                "epoch": ep,
                "train_nll": float(train_nll),
                "train_ppl": float(train_ppl),
                "dev_ppl": float(dev_ppl),
                "epoch_time_s": float(epoch_time),
            }
        )

        if not (math.isfinite(dev_ppl)):
            nan_or_inf = True
            epoch_hist[-1]["note"] = "stopped: dev_ppl_nonfinite"
            break

        if dev_ppl < best_dev:
            best_dev = dev_ppl
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_dev_ppl": float(best_dev) if math.isfinite(best_dev) else None,
        "best_epoch": int(best_epoch),
        "epoch_hist": epoch_hist,
        "grad_norm_hist": grad_norm_hist,
        "nan_or_inf": bool(nan_or_inf),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="directory containing train.csv/dev.csv/test.csv")
    ap.add_argument("--vocab_json", default="result_4_300d/vocab.json", help="Part I vocab.json")
    ap.add_argument("--outdir", default="part2_diff_seed", help="output directory for results")  #part2_runs_300d
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2002)

    # model hparams
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--emb_dim", type=int, default=300)
    ap.add_argument("--hid_dim", type=int, default=300)
    ap.add_argument("--rnn_layers", type=int, default=1)
    ap.add_argument("--tf_layers", type=int, default=4)
    ap.add_argument("--tf_heads", type=int, default=4)
    ap.add_argument("--tf_ff", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--lr_rnn", type=float, default=1e-3)
    ap.add_argument("--lr_tf", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=200)

    ap.add_argument(
        "--emb_mode",
        choices=["scratch_trainable", "self_fixed", "public_fixed", "public_finetune"],
        default="public_finetune",
    )
    ap.add_argument("--emb_path", default="wiki-news-300d-1M.vec", help="embedding text file (.txt/.vec) OR aligned .pt from align_vectors_to_vocab.py")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    splits = load_fixed_splits(args.data_dir, include_context=True)
    if splits is None:
        raise ValueError("Cannot find train.csv/dev.csv/test.csv in data_dir")
    train_df, dev_df, test_df = splits

    itos = load_vocab_json(args.vocab_json)
    stoi = {t: i for i, t in enumerate(itos)}
    vocab = Vocab(stoi=stoi, itos=itos, pad=stoi["<pad>"], unk=stoi["<unk>"], bos=stoi["<bos>"], eos=stoi["<eos>"])

    train_stream = build_stream_from_df(train_df, vocab)
    dev_stream = build_stream_from_df(dev_df, vocab)
    test_stream = build_stream_from_df(test_df, vocab)

    train_data = batchify(train_stream, args.batch_size, device)
    dev_data = batchify(dev_stream, args.batch_size, device)
    test_data = batchify(test_stream, args.batch_size, device)

    emb_matrix_np = None
    emb_weight_torch = None
    coverage = None

    if args.emb_mode != "scratch_trainable":
        if not args.emb_path:
            raise ValueError("emb_path is required for self_fixed/public_fixed/public_finetune")

        if args.emb_path.lower().endswith(".pt"):
            obj = torch.load(args.emb_path, map_location="cpu")
            if "weight" not in obj:
                raise ValueError("Aligned .pt must contain key 'weight'")
            emb_weight_torch = obj["weight"].float().contiguous()
            emb_dim_inferred = int(emb_weight_torch.shape[1])
            if emb_dim_inferred != args.emb_dim:
                raise ValueError(f"Embedding dim mismatch: pt has {emb_dim_inferred}, emb_dim arg is {args.emb_dim}")
            if emb_weight_torch.shape[0] != len(itos):
                raise ValueError(f"Vocab size mismatch: pt has {emb_weight_torch.shape[0]}, vocab has {len(itos)}")
            coverage = float(obj.get("coverage", 0.0))
        else:
            vecs = load_text_vectors(args.emb_path)
            emb_dim_inferred = len(next(iter(vecs.values())))
            if emb_dim_inferred != args.emb_dim:
                raise ValueError(f"Vector dim ({emb_dim_inferred}) != emb_dim ({args.emb_dim}). Make them consistent.")
            emb_matrix_np, coverage = build_embedding_matrix(itos, vecs, dim=args.emb_dim, seed=args.seed)

    def maybe_inject(model: nn.Module):
        if args.emb_mode == "scratch_trainable":
            return
        freeze = args.emb_mode in ["self_fixed", "public_fixed"]
        if emb_weight_torch is not None:
            inject_pretrained_weight_torch(model, emb_weight_torch, freeze=freeze)
        else:
            inject_pretrained_embeddings(model, emb_matrix_np, freeze=freeze)

    results: Dict[str, Any] = {
        "emb_mode": args.emb_mode,
        "emb_path": args.emb_path,
        "coverage": coverage,
        "seed": args.seed,
        "device": str(device),
        "hparams": {
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "dropout": args.dropout,
            "emb_dim": args.emb_dim,
            "hid_dim": args.hid_dim,
            "rnn_layers": args.rnn_layers,
            "tf_layers": args.tf_layers,
            "tf_heads": args.tf_heads,
            "tf_ff": args.tf_ff,
            "max_len": args.max_len,
            "lr_rnn": args.lr_rnn,
            "lr_tf": args.lr_tf,
            "clip": args.clip,
            "log_every": args.log_every,
        },
    }

    # RNN
    rnn = RNNLM(len(itos), args.emb_dim, args.hid_dim, args.rnn_layers, args.dropout).to(device)
    maybe_inject(rnn)
    t0 = time.time()
    rnn_train = train_neural_track(rnn, train_data, dev_data, args.seq_len, args.epochs, args.lr_rnn, args.clip, device, log_every=args.log_every)
    tr_time = time.time() - t0
    results["rnn"] = {
        **rnn_train,
        "dev_ppl": float(eval_ppl_neural(rnn, dev_data, args.seq_len, device)) if not rnn_train.get("nan_or_inf") else None,
        "test_ppl": float(eval_ppl_neural(rnn, test_data, args.seq_len, device)) if not rnn_train.get("nan_or_inf") else None,
        "train_time_s": float(tr_time),
    }

    # LSTM
    lstm = LSTMLM(len(itos), args.emb_dim, args.hid_dim, args.rnn_layers, args.dropout).to(device)
    maybe_inject(lstm)
    t0 = time.time()
    lstm_train = train_neural_track(lstm, train_data, dev_data, args.seq_len, args.epochs, args.lr_rnn, args.clip, device, log_every=args.log_every)
    tr_time = time.time() - t0
    results["lstm"] = {
        **lstm_train,
        "dev_ppl": float(eval_ppl_neural(lstm, dev_data, args.seq_len, device)) if not lstm_train.get("nan_or_inf") else None,
        "test_ppl": float(eval_ppl_neural(lstm, test_data, args.seq_len, device)) if not lstm_train.get("nan_or_inf") else None,
        "train_time_s": float(tr_time),
    }

    # Transformer
    tf = TransformerLM(
        vocab_size=len(itos),
        d_model=args.emb_dim,
        nhead=args.tf_heads,
        num_layers=args.tf_layers,
        dim_ff=args.tf_ff,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)
    maybe_inject(tf)
    t0 = time.time()
    tf_train = train_neural_track(tf, train_data, dev_data, args.seq_len, args.epochs, args.lr_tf, args.clip, device, log_every=args.log_every)
    tr_time = time.time() - t0
    results["transformer"] = {
        **tf_train,
        "dev_ppl": float(eval_ppl_neural(tf, dev_data, args.seq_len, device)) if not tf_train.get("nan_or_inf") else None,
        "test_ppl": float(eval_ppl_neural(tf, test_data, args.seq_len, device)) if not tf_train.get("nan_or_inf") else None,
        "train_time_s": float(tr_time),
    }

    #out_path = outdir / f"part2_results_{args.emb_mode}.json"
    import re

    def _slug(s: str) -> str:
        # keep letters/numbers/_- only, replace others with _
        return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")

    suffix_parts = [args.emb_mode]

    # include emb source name when emb_path is provided (e.g., wiki-news-300d-1M / wiki_giga_aligned_embeddings)
    if getattr(args, "emb_path", None):
        p = Path(args.emb_path)
        suffix_parts.append(_slug(p.stem))   # stem = filename without extension

    # include seed if available (recommended)
    if hasattr(args, "seed") and args.seed is not None:
        suffix_parts.append(f"seed{args.seed}")

    suffix = "__".join(suffix_parts)
    out_path = outdir / f"part2_results_{suffix}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] saved: {out_path}")


if __name__ == "__main__":
    main()
