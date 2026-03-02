# part2/train_word2vec.py
import argparse
from pathlib import Path
import pandas as pd
from common_part1_imports import ensure_text_clean_from_qa
from gensim.models import Word2Vec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/train.csv", help="path to train.csv")
    ap.add_argument("--out", default="part2_self_w2v_300d.vec", help="output vectors (text)")
    ap.add_argument("--dim", type=int, default=300)  
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--sg", type=int, default=1, help="1=skipgram, 0=cbow")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    df = ensure_text_clean_from_qa(df)

    # NOTE: keep consistent with Part I tokenize() behavior (lower + split by whitespace)
    sentences = [str(x).strip().lower().split() for x in df["text_clean"].tolist() if isinstance(x, str)]

    w2v = Word2Vec(
        sentences=sentences,
        vector_size=args.dim,
        window=args.window,
        min_count=args.min_count,
        sg=args.sg,
        workers=args.workers,
    )

    out_path = Path(args.out)
    w2v.wv.save_word2vec_format(str(out_path), binary=False)
    print(f"[DONE] saved to {out_path}")

if __name__ == "__main__":
    main()