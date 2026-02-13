#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean ePQA candidate text (all sources) into a corpus usable for LM training.

Outputs:
  1) cleaned_candidates.parquet  (id, qid, asin, source, text_clean)
  2) corpus.jsonl               (one json per line: {id, source, text})
  3) corpus.txt                 (one document per line)

Assumptions:
- Your input file is a CSV (or TSV) with at least:
    - source column: one of {attribute, bullet, cqa, description, review}
    - candidate text column: candidate (default) OR evidence/text/passsage
- Optional columns: qid, ASIN, id (if missing, will be created)

Usage examples:
  python clean_epqa_candidates.py --input epqa_train.csv --outdir out
  python clean_epqa_candidates.py --input epqa.csv --sep $'\t' --text-col candidate --source-col source
"""

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

# -------------------------
# Regex patterns (global)
# -------------------------
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
RE_LONG_NUM = re.compile(r"\b\d{5,}\b")  # long numeric strings (order ids, etc.)
RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_REPEAT_PUNCT = re.compile(r"([!?])\1{2,}")
RE_DOTS = re.compile(r"\.{4,}")

# review meta lines (common)
RE_REVIEW_META_LINES = re.compile(
    r"(?im)^\s*(verified purchase|helpful|report|reviewed in|rating|stars?:?)\b.*$"
)

# bullets
RE_BULLET_PREFIX = re.compile(r"(?m)^\s*([•·●▪️▫️\-–—\*]+)\s*")

# attribute key-value patterns
RE_ATTR_SEP = re.compile(r"\s*(?:[:=\-–—]+)\s*")

# normalize quotes/dashes optionally
TRANSLATE_TABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",  # nbsp
    }
)

ALLOWED_SOURCES = {"attribute", "bullet", "cqa", "description", "review"}


def normalize_unicode(text: str) -> str:
    # NFKC + normalize common punctuation
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(TRANSLATE_TABLE)
    return text


def strip_html(text: str) -> str:
    # unescape html entities first, then remove tags
    text = html.unescape(text)
    text = RE_HTML_TAG.sub(" ", text)
    return text


def standardize_whitespace(text: str) -> str:
    # Normalize newlines, collapse spaces, keep paragraph structure
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # remove excessive newlines but keep at most 2
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    # collapse spaces/tabs but not newlines
    text = "\n".join(RE_MULTI_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    # drop empty leading/trailing newlines
    text = text.strip()
    return text


def replace_sensitive_tokens(text: str) -> str:
    text = RE_URL.sub("<URL>", text)
    text = RE_EMAIL.sub("<EMAIL>", text)
    # replace long numbers but keep shorter like 16GB, 2.5, etc.
    text = RE_LONG_NUM.sub("<NUM>", text)
    # normalize excessive punctuation
    text = RE_REPEAT_PUNCT.sub(r"\1", text)
    text = RE_DOTS.sub("...", text)
    return text


def english_ratio(text: str) -> float:
    # Simple heuristic: ratio of ASCII letters among non-space chars.
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    letters = sum(1 for c in chars if ("a" <= c.lower() <= "z"))
    return letters / len(chars)


def token_count_rough(text: str) -> int:
    # rough word count
    return len(re.findall(r"\b\w+\b", text))


def clean_attribute(text: str) -> str:
    """
    Turn key-value-ish attribute strings into readable sentences.
    Examples:
      "Color: Black" -> "Color: Black."
      "Material - Cotton" -> "Material: Cotton."
    If multiple lines, process line-by-line.
    """
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        # common junk values
        if line.lower() in {"n/a", "na", "none", "unknown", "not applicable"}:
            continue
        # If already contains ':', keep; else try to normalize separators
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
        else:
            parts = RE_ATTR_SEP.split(line, maxsplit=1)
            if len(parts) == 2:
                key, val = parts[0].strip(), parts[1].strip()
            else:
                # no clear key-value; keep as-is
                lines.append(line if line.endswith((".", "!", "?")) else (line + "."))
                continue

        # drop empty/meaningless values
        if not val or val.lower() in {"n/a", "na", "none", "unknown", "not applicable"}:
            continue

        sent = f"{key}: {val}"
        if not sent.endswith((".", "!", "?")):
            sent += "."
        lines.append(sent)

    return "\n".join(lines).strip()


def clean_bullet(text: str) -> str:
    """
    Normalize bullets to "- " per line and add sentence-ending punctuation if missing.
    """
    out_lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        line = RE_BULLET_PREFIX.sub("- ", line)
        # If line starts without "- ", keep; else ensure single "- "
        if line.startswith("- "):
            content = line[2:].strip()
            if content and not content.endswith((".", "!", "?")):
                content += "."
            out_lines.append("- " + content)
        else:
            # not a bullet line; treat as sentence
            if not line.endswith((".", "!", "?")):
                line += "."
            out_lines.append(line)
    return "\n".join(out_lines).strip()

'''
def clean_cqa(text: str) -> str:
    """
    Preserve QA structure if present; normalize Q/A prefixes.
    """
    # normalize common prefixes
    text = re.sub(r"(?im)^\s*(question|q)\s*[:\-]\s*", "Q: ", text)
    text = re.sub(r"(?im)^\s*(answer|a)\s*[:\-]\s*", "A: ", text)
    # If it is a single question sentence with '?', keep as is
    return text.strip()
'''
RE_ANSWER_THEN_QUESTION = re.compile(r"(?is)^\s*(.+?)\s*Question:\s*(.+?)\s*$")

def clean_cqa(text: str) -> str:
    """
    Convert: "<answer> Question: <question>"
    Into:    "Q: <question>\nA: <answer>"
    """
    text = text.strip()

    m = RE_ANSWER_THEN_QUESTION.match(text)
    if m:
        ans = m.group(1).strip(" \n\t-:")
        q = m.group(2).strip()

        # normalize punctuation
        if q and not q.endswith(("?", ".", "!")):
            q += "?"
        if ans and not ans.endswith((".", "!", "?")):
            ans += "."

        return f"Q: {q}\nA: {ans}"

    # fallback: keep as-is
    return text


def clean_description(text: str) -> str:
    # mostly global cleaning already; keep paragraphs
    return text.strip()


def clean_review(text: str) -> str:
    """
    Remove common metadata lines; keep the rest.
    """
    # Remove meta lines
    text = RE_REVIEW_META_LINES.sub("", text)
    # Remove stray emoji (optional; keep ascii)
    text = "".join(ch for ch in text if ord(ch) < 128 or ch in {"\n", "\t"})
    text = standardize_whitespace(text)
    return text.strip()


def source_specific_cleanup(text: str, source: str) -> str:
    source = (source or "").strip().lower()
    if source == "attribute":
        return clean_attribute(text)
    if source == "bullet":
        return clean_bullet(text)
    if source == "cqa":
        return clean_cqa(text)
    if source == "description":
        return clean_description(text)
    if source == "review":
        return clean_review(text)
    # unknown -> return unchanged
    return text.strip()


def infer_text_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    candidates = ["candidate", "evidence", "text", "passage", "snippet", "content"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot infer text column. Available columns: {list(df.columns)}")


def infer_source_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in ["source", "cand_source", "candidate_source"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot infer source column. Available columns: {list(df.columns)}")


def ensure_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        df["id"] = [f"row_{i}" for i in range(len(df))]
    # normalize common column names for convenience
    if "ASIN" not in df.columns and "asin" in df.columns:
        df["ASIN"] = df["asin"]
    if "qid" not in df.columns and "QID" in df.columns:
        df["qid"] = df["QID"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="train_raw.csv", help="Input CSV/TSV file path for ePQA.")
    ap.add_argument("--sep", default=",", help="CSV separator, default ','; use '\\t' for TSV.")
    ap.add_argument("--outdir", default="processed_data", help="Output directory.")
    ap.add_argument("--text-col", default=None, help="Text column name (default: auto).")
    ap.add_argument("--source-col", default=None, help="Source column name (default: auto).")
    ap.add_argument("--min-words", type=int, default=5, help="Drop rows with fewer words after cleaning.")
    ap.add_argument("--min-eng-ratio", type=float, default=0.60, help="Drop rows with english_ratio < threshold.")
    ap.add_argument("--max-words", type=int, default=400, help="Truncate documents to this many words (0 disables).")
    ap.add_argument("--dedup", action="store_true", help="Exact dedup on cleaned text (recommended).")
    args = ap.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, sep=args.sep, dtype=str, keep_default_na=False)
    df = ensure_id_cols(df)

    text_col = infer_text_col(df, args.text_col)
    source_col = infer_source_col(df, args.source_col)

    # Normalize source
    df[source_col] = df[source_col].astype(str).str.strip().str.lower()

    # Optionally keep only known sources (but you said full set; just warn)
    unknown_sources = sorted(set(df[source_col].unique()) - ALLOWED_SOURCES)
    if unknown_sources:
        print(f"[WARN] Unknown sources found: {unknown_sources}. They will be kept but cleaned with global rules only.")

    cleaned_rows = []
    for _, row in df.iterrows():
        src = str(row.get(source_col, "")).strip().lower()
        txt = str(row.get(text_col, "")).strip()
        if not txt:
            continue

        # Global cleaning
        txt = normalize_unicode(txt)
        txt = strip_html(txt)
        txt = replace_sensitive_tokens(txt)
        txt = standardize_whitespace(txt)

        # Source-specific
        txt = source_specific_cleanup(txt, src)

        # Re-apply global whitespace normalization after source cleanup
        txt = standardize_whitespace(txt)

        # Filters
        if english_ratio(txt) < args.min_eng_ratio:
            continue
        if token_count_rough(txt) < args.min_words:
            continue

        # Truncate (word-based) to keep training stable
        if args.max_words and args.max_words > 0:
            words = txt.split()
            if len(words) > args.max_words:
                txt = " ".join(words[: args.max_words]).strip()

        cleaned_rows.append(
            {
                "id": row.get("id"),
                "qid": row.get("qid", ""),
                "ASIN": row.get("ASIN", ""),
                "source": src,
                "text_clean": txt,
            }
        )

    out = pd.DataFrame(cleaned_rows)

    # Exact dedup on cleaned text (keeps first occurrence)
    if args.dedup and not out.empty:
        before = len(out)
        out = out.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
        after = len(out)
        print(f"[INFO] Exact dedup: {before} -> {after} (removed {before-after})")

    # Save cleaned table
    parquet_path = outdir / "cleaned_candidates.parquet"
    out.to_parquet(parquet_path, index=False)

    # Save JSONL corpus
    jsonl_path = outdir / "corpus.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in out.itertuples(index=False):
            rec = {"id": r.id, "source": r.source, "text": r.text_clean}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save TXT corpus (one doc per line)
    txt_path = outdir / "corpus.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for t in out["text_clean"].tolist():
            # ensure single-line doc in corpus.txt
            f.write(t.replace("\n", "\\n") + "\n")

    # Simple summary
    print(f"[DONE] Cleaned rows: {len(out)}")
    if len(out):
        print("[INFO] Source distribution after cleaning:")
        print(out["source"].value_counts())
    print(f"[OUT] {parquet_path}")
    print(f"[OUT] {jsonl_path}")
    print(f"[OUT] {txt_path}")


if __name__ == "__main__":
    main()
