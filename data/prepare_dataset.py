"""
Dataset preparation pipeline for AI-validator-service toxicity fine-tuning.
Output schema: text (str), label (int8), lang (str), source (str), category (str)
"""
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

OUT      = Path("data/processed")
DATA_DIR = Path("data/datasets")         
OUT.mkdir(parents=True, exist_ok=True)

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


# ── 1. Civil Comments ──────────────────────────────────
def load_civil_comments() -> pd.DataFrame:
    ds = load_dataset("civil_comments", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    label = (df["toxicity"] >= 0.5).astype("int8")
    return base_frame(df["text"], label, "en", "civil_comments")


# ── 2. Multilingual parquet shards ────────────────────────────────────────────
def _label_from_parquet(df: pd.DataFrame) -> pd.Series:
    for col in ("toxic", "toxicity", "label", "is_toxic", "target", "inappropriate"):
        if col in df.columns:
            raw = df[col]
            if raw.dtype in ("int8", "int16", "int32", "int64", "bool"):
                return raw.astype("int8")
            if pd.api.types.is_float_dtype(raw):
                return (raw >= 0.5).astype("int8")
            try:
                return (pd.to_numeric(raw, errors="coerce").fillna(0.0) >= 0.5).astype("int8")
            except Exception as exc:
                raise KeyError(
                    f"Column '{col}' found but could not be cast to float: {exc}. "
                    f"Sample values: {raw.dropna().head(5).tolist()}"
                )
    raise KeyError(
        f"No known label column found. Available columns: {list(df.columns)}"
    )


def _text_col(df: pd.DataFrame) -> pd.Series:
    for col in ("text", "comment_text", "content", "message", "sentence"):
        if col in df.columns:
            return df[col]
    raise KeyError(
        f"No known text column found. Available columns: {list(df.columns)}"
    )


def load_parquet_shards() -> pd.DataFrame:
    frames = []
    for prefix, lang_code in PARQUET_LANGS.items():
        pattern = f"{prefix}-00000-of-00001.parquet"
        path = DATA_DIR / pattern
        if not path.exists():
            print(f"    [skip] {pattern} not found")
            continue
        try:
            raw = pd.read_parquet(path)
            text  = _text_col(raw)
            label = _label_from_parquet(raw)
            df = base_frame(text, label, lang_code, f"parquet_{prefix}")
            frames.append(df)
            print(f"    {pattern}: {len(df):,} rows  (lang={lang_code})")
        except Exception as exc:
            print(f"    [FAILED] {pattern}: {exc}")

    if not frames:
        raise RuntimeError("No parquet shards loaded — check DATA_DIR path.")
    return pd.concat(frames, ignore_index=True)


# ── 3. Inappappropriate_messages.csv ──────────────────────────────────────────
def load_inappropriate_messages() -> pd.DataFrame:
    csv_path = DATA_DIR / "Inappapropriate_messages.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")
 
    raw = pd.read_csv(csv_path)
    print(f"    CSV columns: {list(raw.columns)}")
 
    text = _text_col(raw)
 
    try:
        label = _label_from_parquet(raw)          
    except KeyError:
        print("    No label column in CSV — treating all rows as toxic (1)")
        label = pd.Series([1] * len(raw), dtype="int8")
 
    lang_col = next((c for c in raw.columns if c.lower() in ("lang", "language", "locale")), None)
    if lang_col:
        lang = raw[lang_col].fillna("ru").astype(str)
    else:
        lang = "ru"   
 
    return base_frame(text, label, lang, "inappropriate_messages", "inappropriate")

# ── pipeline ──────────────────────────────────────────────────────────────────
def run() -> None:
    print("Loading datasets...\n")

    frames: list[pd.DataFrame] = []

    # --- HuggingFace datasets ---
    for loader in [load_civil_comments]:
        try:
            df = loader()
            print(f"  {loader.__name__}: {len(df):,} rows")
            frames.append(df)
        except Exception as exc:
            print(f"  {loader.__name__} FAILED: {exc}")

    # --- Local parquet shards ---
    print("\n  Loading parquet shards:")
    try:
        df = load_parquet_shards()
        print(f"  → parquet total: {len(df):,} rows")
        frames.append(df)
    except Exception as exc:
        print(f"  Parquet shards FAILED: {exc}")

    # --- CSV ---
    print("\n  Loading CSV:")
    try:
        df = load_inappropriate_messages()
        print(f"  Inappappropriate_messages.csv: {len(df):,} rows")
        frames.append(df)
    except Exception as exc:
        print(f"  CSV FAILED: {exc}")

    # ── combine ───────────────────────────────────────────────────────────────
    if not frames:
        raise RuntimeError("All loaders failed — nothing to process.")

    df = pd.concat(frames, ignore_index=True)
    print(f"\nRaw combined: {len(df):,} rows")

    # ── quality filter ────────────────────────────────────────────────────────
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len().between(3, 512)]
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype("int8")

    # ── deduplicate ───────────────────────────────────────────────────────────
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