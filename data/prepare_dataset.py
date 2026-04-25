"""
Dataset preparation pipeline for AI-validator-service toxicity fine-tuning.
Output schema: text (str), label (int8), lang (str), source (str), category (str)

All datasets are pulled from HuggingFace: esclient/toxicity_multilanguage_dataset
"""
from pyexpat import model

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from pathlib import Path
from augment_dataset import augment_and_combine

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

HF_REPO_ID   = "esclient/toxicity_multilanguage_dataset"
HF_REPO_TYPE = "dataset"

# Language code → ISO 639-1 tag for the parquet shards
PARQUET_LANGS: dict[str, str] = {
    "am":  "am",   # Amharic
    "ar":  "ar",   # Arabic
    "de":  "de",   # German
    "en":  "en",   # English
    "es":  "es",   # Spanish
    "fr":  "fr",   # French
    "he":  "he",   # Hebrew
    "hi":  "hi",   # Hindi
    "hin": "hi",   # Hindi (alternate shard, same ISO code)
    "it":  "it",   # Italian
    "ja":  "ja",   # Japanese
    "ru":  "ru",   # Russian
    "tt":  "tt",   # Tatar
    "uk":  "uk",   # Ukrainian
    "zh":  "zh",   # Chinese
}


def _hf_download(filename: str) -> str:
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type=HF_REPO_TYPE,
    )


def base_frame(
    text,
    label,
    lang: str,
    source: str,
    category: str = "general",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text":     pd.Series(text, dtype="object"),
            "label":    pd.Series(label, dtype="int8"),
            "lang":     lang,
            "source":   source,
            "category": category,
        }
    )

def _to_binary_label(raw: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(raw, errors="coerce").fillna(0.0)
    return (numeric >= 0.5).astype("int8")


def _label_from_parquet(df: pd.DataFrame) -> pd.Series:
    for col in ("toxic", "toxicity", "label", "is_toxic", "target", "inappropriate"):
        if col in df.columns:
            raw = df[col]
            try:
                return _to_binary_label(raw)
            except Exception as exc:
                raise KeyError(
                    f"Column '{col}' found but could not be cast to float: {exc}. "
                    f"Sample values: {raw.dropna().head(5).tolist()}"
                )
    raise KeyError(
        f"No known label column found. Available columns: {list(df.columns)}"
    )


def _text_col(df: pd.DataFrame) -> pd.Series:
    for col in ("text", "comment", "comments", "comment_text", "content", "message", "sentence"):
        if col in df.columns:
            return df[col]
    raise KeyError(
        f"No known text column found. Available columns: {list(df.columns)}"
    )


# ── 1. Civil Comments ──────────────────────────────────────────────────────────
def load_civil_comments() -> pd.DataFrame:
    ds = load_dataset("civil_comments", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    label = _to_binary_label(df["toxicity"])
    return base_frame(df["text"], label, "en", "civil_comments")


# ── 2. Multilingual parquet shards ────────────────────────────────────────────
def load_parquet_shards() -> pd.DataFrame:
    frames = []
    for prefix, lang_code in PARQUET_LANGS.items():
        filename = f"{prefix}-00000-of-00001.parquet"
        try:
            local_path = _hf_download(filename)
            raw   = pd.read_parquet(local_path)
            text  = _text_col(raw)
            label = _label_from_parquet(raw)
            df    = base_frame(text, label, lang_code, f"parquet_{prefix}")
            frames.append(df)
            print(f"    {filename}: {len(df):,} rows  (lang={lang_code})")
        except Exception as exc:
            print(f"    [FAILED] {filename}: {exc}")

    if not frames:
        raise RuntimeError("No parquet shards loaded — check HF repo access.")
    return pd.concat(frames, ignore_index=True)


# ── 3. Inappappropriate_messages.csv ──────────────────────────────────────────
def load_inappropriate_messages() -> pd.DataFrame:
    local_path = _hf_download("Inappapropriate_messages.csv")
    raw = pd.read_csv(local_path)
    print(f"    CSV columns: {list(raw.columns)}")

    text = _text_col(raw)

    try:
        label = _label_from_parquet(raw)
    except KeyError:
        print("    No label column in CSV — treating all rows as toxic (1)")
        label = pd.Series([1] * len(raw), dtype="int8")

    lang_col = next((c for c in raw.columns if c.lower() in ("lang", "language", "locale")), None)
    lang = raw[lang_col].fillna("ru").astype(str) if lang_col else "ru"

    return base_frame(text, label, lang, "inappropriate_messages", "inappropriate")


# ── 4. labled.csv (abusive True/False) ────────────────────────────────────────
def load_labled_csv() -> pd.DataFrame:
    local_path = _hf_download("labled.csv")
    raw = pd.read_csv(local_path)
    print(f"    CSV columns: {list(raw.columns)}")
    text  = _text_col(raw)
    label = _to_binary_label(raw["abusive"].map({"True": 1, "False": 0, True: 1, False: 0}))
    return base_frame(text, label, "ru", "labled_csv")


# ── 5. russian_dataset.jsonl ──────────────────────────────────────────────────
def load_russian_jsonl() -> pd.DataFrame:
    local_path = _hf_download("russian_dataset.jsonl")
    raw = pd.read_json(local_path, lines=True)
    print(f"    JSONL columns: {list(raw.columns)}")
    text  = _text_col(raw)
    label = _label_from_parquet(raw)
    return base_frame(text, label, "ru", "russian_jsonl")


# ── 6. russian_dataset_2.tsv (parallel corpus) ────────────────────────────────
def load_russian_tsv2() -> pd.DataFrame:
    local_path = _hf_download("russian_dataset_2.tsv")
    raw = pd.read_csv(local_path, sep="\t")
    print(f"    TSV columns: {list(raw.columns)}")

    toxic = base_frame(raw["ru_toxic_comment"],
                       pd.Series([1] * len(raw), dtype="int8"),
                       "ru", "russian_tsv2", "general")
    clean = base_frame(raw["ru_neutral_comment"],
                       pd.Series([0] * len(raw), dtype="int8"),
                       "ru", "russian_tsv2", "general")
    return pd.concat([toxic, clean], ignore_index=True)


# ── 7. russian_distorted_toxicity.tsv ─────────────────────────────────────────
def load_russian_distorted() -> pd.DataFrame:
    local_path = _hf_download("russian_distorted_toxicity.tsv")
    raw = pd.read_csv(local_path, sep="\t")
    print(f"    TSV columns: {list(raw.columns)}")

    raw   = raw.dropna(subset=["comments", "toxicity"])
    label = _to_binary_label(raw["toxicity"])
    return base_frame(raw["comments"], label, "ru", "russian_distorted")


# ── 8. russian_comments_from_2ch_pikabu.csv ───────────────────────────────────
def load_russian_comments_2ch_pikabu() -> pd.DataFrame:
    local_path = _hf_download("russian_comments_from_2ch_pikabu.csv")
    raw = pd.read_csv(local_path)
    print(f"    CSV columns: {list(raw.columns)}")
    text  = _text_col(raw)
    label = _label_from_parquet(raw)
    return base_frame(text, label, "ru", "russian_comments_2ch_pikabu")


# ── 9. labeled.csv ────────────────────────────────────────────────────────────
def load_labeled_csv() -> pd.DataFrame:
    local_path = _hf_download("labeled.csv")
    raw = pd.read_csv(local_path)
    print(f"    CSV columns: {list(raw.columns)}")
    text  = _text_col(raw)
    label = _label_from_parquet(raw)
    return base_frame(text, label, "ru", "labeled_csv")


# ── 10. russian_dataset_3.txt (fastText labels) ───────────────────────────────
def load_russian_dataset_3() -> pd.DataFrame:
    local_path = _hf_download("russian_dataset_3.txt")

    rows = []
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            labels_part, text = parts[0], parts[1]
            labels   = [p.strip() for p in labels_part.split(",") if p.strip()]
            is_toxic = any(lbl != "__label__NORMAL" for lbl in labels)
            rows.append((text, 1 if is_toxic else 0))

    raw = pd.DataFrame(rows, columns=["text", "label"])
    return base_frame(raw["text"], _to_binary_label(raw["label"]), "ru", "russian_dataset_3_txt")

def run() -> None:
    print("Loading datasets...\n")

    frames: list[pd.DataFrame] = []

    for loader in [load_civil_comments]:
        try:
            df = loader()
            print(f"  {loader.__name__}: {len(df):,} rows")
            frames.append(df)
        except Exception as exc:
            print(f"  {loader.__name__} FAILED: {exc}")

    print("\n  Loading parquet shards:")
    try:
        df = load_parquet_shards()
        print(f"  → parquet total: {len(df):,} rows")
        frames.append(df)
    except Exception as exc:
        print(f"  Parquet shards FAILED: {exc}")

    print("\n  Loading local files:")
    local_loaders = [
        load_inappropriate_messages,
        load_labled_csv,
        load_labeled_csv,
        load_russian_comments_2ch_pikabu,
        load_russian_jsonl,
        load_russian_dataset_3,
        load_russian_tsv2,
        load_russian_distorted,
    ]
    for loader in local_loaders:
        try:
            df = loader()
            print(f"  {loader.__name__}: {len(df):,} rows")
            frames.append(df)
        except Exception as exc:
            print(f"  {loader.__name__} FAILED: {exc}")

    if not frames:
        raise RuntimeError("All loaders failed — nothing to process.")

    df = pd.concat(frames, ignore_index=True)
    df = augment_and_combine(df)
    print(f"\nRaw combined: {len(df):,} rows")

    # ── quality filter ────────────────────────────────────────────────────────
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len().between(3, 512)]
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype("int8")

    df["_key"] = df["text"].str.lower()
    df = df.drop_duplicates(subset="_key").drop(columns="_key")
    print(f"After dedup + filter: {len(df):,} rows")

    print("\nLabel distribution:")
    print(df["label"].value_counts().rename({0: "clean", 1: "toxic"}))
    print("\nLanguage distribution:")
    print(df["lang"].value_counts())

    # ── stratified 80 / 10 / 10 split ────────────────────────────────────────
    train, tmp = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    val, test = train_test_split(
        tmp, test_size=0.5, stratify=tmp["label"], random_state=42
    )

    for split, name in [(train, "train"), (val, "val"), (test, "test")]:
        path = OUT / f"{name}.parquet"
        split.reset_index(drop=True).to_parquet(path, index=False)
        print(f"Saved {name}: {len(split):,} rows → {path}")


if __name__ == "__main__":
    run()