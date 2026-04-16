"""
Dataset preparation pipeline for AI-validator-service toxicity fine-tuning.
Output schema: text (str), label (int8), lang (str), source (str), category (str)
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def base_frame(text, label, lang, source, category="general"):
    return pd.DataFrame({
        "text":     text,
        "label":    label,
        "lang":     lang,
        "source":   source,
        "category": category,
    })

# ── 1. Civil Comments ─────────────────────────────────────────────────────────
def load_civil_comments() -> pd.DataFrame:
    ds = load_dataset("civil_comments", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    label = (df["toxicity"] >= 0.5).astype("int8")
    return base_frame(df["text"], label, "en", "civil_comments")

# ── 2. Russian Toxic Comments ─────────────────────────────────────
def load_ru_toxic() -> pd.DataFrame:
    ds = load_dataset("skolkovo_institute/russian_toxicity_dataset", split="train")
    df = ds.to_pandas()
    label = df["label"].astype("int8")
    return base_frame(df["text"], label, "ru", "ru_toxic_skoltech")


# ── 3. Russian Inappropriate Messages ────────────────
def load_ru_inappropriate() -> pd.DataFrame:
    ds = load_dataset("NiGuLa/Russian_Inappropriate_Messages", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    label = (df["inappropriate"] >= 0.5).astype("int8")
    return base_frame(df["text"], label, "ru", "ru_inappropriate_apanc")


# ── 4. Jigsaw Multilingual ────────────────────────────────────
def load_jigsaw_multilingual_all() -> pd.DataFrame:
    ds = load_dataset("community-datasets/jigsaw_multilingual_toxic_comment_classification", split="validation")
    df = ds.to_pandas()
    label = (df["toxic"] >= 0.5).astype("int8")
    return base_frame(df["comment_text"], label, "multi", "jigsaw_multilingual")

# ── pipeline ──────────────────────────────────────────────────────────────────


def run():
    print("Loading datasets...")
    frames = []

    for loader in [load_civil_comments, load_ru_toxic, load_ru_inappropriate, load_jigsaw_multilingual_all]:
        try:
            df = loader()
            print(f"  {loader.__name__}: {len(df):,} rows")
            frames.append(df)
        except Exception as e:
            print(f"  {loader.__name__} FAILED: {e}")

    df = pd.concat(frames, ignore_index=True)
    print(f"\nRaw combined: {len(df):,} rows")

    # quality filter
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len().between(3, 512)]
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype("int8")

    # deduplicate
    df["_key"] = df["text"].str.lower()
    df = df.drop_duplicates(subset="_key").drop(columns="_key")
    print(f"After dedup + filter: {len(df):,} rows")
    print(df["label"].value_counts().rename({0: "clean", 1: "toxic"}))

    # stratified 80/10/10 split
    train, tmp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val, test  = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=42)

    for split, name in [(train, "train"), (val, "val"), (test, "test")]:
        path = OUT / f"{name}.parquet"
        split.reset_index(drop=True).to_parquet(path, index=False)
        print(f"Saved {name}: {len(split):,} rows → {path}")


if __name__ == "__main__":
    run()