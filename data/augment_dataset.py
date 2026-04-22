"""
Synthetic dataset augmentation for AI-validator-service toxicity fine-tuning.

Augmentation strategies (all operate on post-normalizer text from moderation-service):
  - synthetic_concat         : full space removal between all words
  - synthetic_partial_concat : priority-weighted space removal (toxic boundaries > adjacent > random)
  - synthetic_context_filler : toxic sentence + neutral filler interleaved
  - synthetic_context_mix    : toxic + clean concatenation, label derived from toxic ratio
  - synthetic_crosslang      : toxic sentences from different languages smashed together

Label logic for synthetic_context_mix:
  toxic_ratio >= 0.70  -> label 1  (category: "concatenation")
  toxic_ratio  0.30-0.70 -> label 0  (category: "context_ambiguous")
  toxic_ratio <= 0.30  -> label 0  (category: "context")

Volume control:
  - Synthetic rows fill gaps to balance (category, label) buckets
  - Synthetic rows are hard-capped at MAX_SYNTHETIC_RATIO of total dataset
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ── constants ────────────────────────────────────────────────────────────────

MAX_SYNTHETIC_RATIO = 0.40   # synthetic rows never exceed 40% of final dataset

# Priority weights for partial-concat boundary removal
P_TOXIC_BOUNDARY    = 0.85   # space between toxic word and any neighbour
P_ADJACENT_BOUNDARY = 0.50   # space one hop away from a toxic word
P_RANDOM_BOUNDARY   = 0.15   # all other word boundaries

# Context-mix label thresholds
TOXIC_RATIO_HIGH = 0.70
TOXIC_RATIO_LOW  = 0.30

# How many toxic words to extract via TF-IDF
TOP_N_TOXIC_WORDS = 300

# Seed for reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ── toxic vocabulary bootstrap ────────────────────────────────────────────────

def build_toxic_vocabulary(df: pd.DataFrame, top_n: int = TOP_N_TOXIC_WORDS) -> set[str]:
    """
    Bootstrap a toxic-word vocabulary from the combined dataframe using TF-IDF
    discrimination between label=1 and label=0 texts.
    Words that score highest in toxic docs relative to clean docs are selected.
    """
    toxic_texts = df[df["label"] == 1]["text"].astype(str).tolist()
    clean_texts = df[df["label"] == 0]["text"].astype(str).tolist()

    if not toxic_texts or not clean_texts:
        return set()

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
        min_df=3,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    all_texts = toxic_texts + clean_texts
    labels    = [1] * len(toxic_texts) + [0] * len(clean_texts)

    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    toxic_mask = np.array(labels) == 1
    clean_mask = ~toxic_mask

    toxic_mean = np.asarray(tfidf_matrix[toxic_mask].mean(axis=0)).flatten()
    clean_mean = np.asarray(tfidf_matrix[clean_mask].mean(axis=0)).flatten()

    discrimination = toxic_mean - clean_mean
    top_indices    = np.argsort(discrimination)[::-1][:top_n]
    toxic_vocab    = set(feature_names[top_indices].tolist())

    print(f"  [vocab] Bootstrapped {len(toxic_vocab)} toxic indicator words via TF-IDF")
    return toxic_vocab


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return text.split()


def _is_toxic_word(word: str, vocab: set[str]) -> bool:
    return word.lower().strip(".,!?;:\"'") in vocab


def _classify_boundaries(tokens: list[str], vocab: set[str]) -> list[str]:
    n = len(tokens)
    if n <= 1:
        return []

    is_toxic = [_is_toxic_word(t, vocab) for t in tokens]
    classes  = []

    for i in range(n - 1):
        left_toxic  = is_toxic[i]
        right_toxic = is_toxic[i + 1]
        left_adj    = (i > 0     and is_toxic[i - 1])
        right_adj   = (i + 2 < n and is_toxic[i + 2])

        if left_toxic or right_toxic:
            classes.append("toxic")
        elif left_adj or right_adj:
            classes.append("adjacent")
        else:
            classes.append("random")

    return classes


def _remove_space(boundary_class: str) -> bool:
    p = {
        "toxic":    P_TOXIC_BOUNDARY,
        "adjacent": P_ADJACENT_BOUNDARY,
        "random":   P_RANDOM_BOUNDARY,
    }[boundary_class]
    return random.random() < p


# ── augmentation functions ─────────────────────────────────────────────────────

def aug_full_concat(text: str, **_) -> str:
    return "".join(_tokenize(text))


def aug_partial_concat(text: str, vocab: set[str], **_) -> str:
    tokens = _tokenize(text)
    if len(tokens) <= 1:
        return text

    classes = _classify_boundaries(tokens, vocab)
    result  = tokens[0]
    for i, cls in enumerate(classes):
        sep     = "" if _remove_space(cls) else " "
        result += sep + tokens[i + 1]
    return result


def aug_context_filler(
    text: str,
    filler_pool: list[str],
    n_fillers: int = 1,
    **_,
) -> str:
    fillers  = random.sample(filler_pool, min(n_fillers, len(filler_pool)))
    parts    = fillers + [text]
    random.shuffle(parts)
    return " ".join(p.strip().rstrip(".") + "." for p in parts)


def aug_crosslang_concat(
    text: str,
    lang: str,
    lang_pool: dict[str, list[str]],
    **_,
) -> str | None:
    other_langs = [l for l in lang_pool if l != lang and lang_pool[l]]
    if not other_langs:
        return None
    other_lang = random.choice(other_langs)
    other_text = random.choice(lang_pool[other_lang])
    parts      = [text.strip(), other_text.strip()]
    random.shuffle(parts)
    sep = "" if random.random() < 0.5 else " "
    return sep.join(parts)


# ── context mix label logic ───────────────────────────────────────────────────

def _context_mix_label_and_category(toxic_ratio: float) -> tuple[int, str]:
    if toxic_ratio >= TOXIC_RATIO_HIGH:
        return 1, "concatenation"
    elif toxic_ratio <= TOXIC_RATIO_LOW:
        return 0, "context"
    else:
        return 0, "context_ambiguous"


def _build_context_mix_row(
    toxic_text: str,
    clean_text: str,
) -> tuple[str, int, str]:
    t_len = len(toxic_text)
    c_len = len(clean_text)
    total = t_len + c_len
    if total == 0:
        return toxic_text, 1, "concatenation"

    toxic_ratio = t_len / total
    label, category = _context_mix_label_and_category(toxic_ratio)

    parts = [toxic_text.strip(), clean_text.strip()]
    random.shuffle(parts)
    sep  = "" if random.random() < 0.5 else " "
    text = sep.join(parts)
    return text, label, category


# ── volume balancer ───────────────────────────────────────────────────────────

def _target_counts(
    real_df: pd.DataFrame,
    buckets: list[tuple[str, int]],
) -> dict[tuple[str, int], int]:
    """
    Compute how many synthetic rows to generate per (category, label) bucket
    so all buckets are balanced, without exceeding MAX_SYNTHETIC_RATIO overall.
    """
    real_counts: dict[tuple[str, int], int] = defaultdict(int)
    for cat, lbl in buckets:
        mask = (real_df["category"] == cat) & (real_df["label"] == lbl)
        real_counts[(cat, lbl)] = int(mask.sum())

    max_real  = max(real_counts.values()) if real_counts else 0
    targets   = {}
    for key, cnt in real_counts.items():
        targets[key] = max(0, max_real - cnt)

    total_real      = len(real_df)
    total_synthetic = sum(targets.values())
    max_allowed     = int(total_real * MAX_SYNTHETIC_RATIO / (1 - MAX_SYNTHETIC_RATIO))

    if total_synthetic > max_allowed and total_synthetic > 0:
        scale = max_allowed / total_synthetic
        targets = {k: int(v * scale) for k, v in targets.items()}

    return targets


# ── main augmentation pipeline ────────────────────────────────────────────────

def build_augmented_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = (pd.to_numeric(df["label"], errors="coerce").fillna(0.0) >= 0.5).astype("int8")

    print("\n[augment] Building toxic vocabulary...")
    vocab = build_toxic_vocabulary(df)

    toxic_df = df[df["label"] == 1].copy()
    clean_df = df[df["label"] == 0].copy()

    toxic_texts = toxic_df["text"].astype(str).tolist()
    clean_texts = clean_df["text"].astype(str).tolist()

    lang_pool: dict[str, list[str]] = defaultdict(list)
    for _, row in toxic_df.iterrows():
        lang_pool[row["lang"]].append(row["text"])

    filler_pool = clean_texts  

    # ── figure out how many rows we need per bucket ───────────────────────────
    buckets = [
        ("concatenation",    0),
        ("concatenation",    1),
        ("context",          0),
        ("context",          1),
        ("context_ambiguous",0),
        ("context_ambiguous",1),
        ("general",          0),
        ("general",          1),
    ]
    targets = _target_counts(df, buckets)
    print(f"[augment] Synthetic targets per bucket: {dict(targets)}")

    rows: list[dict] = []

    def add_row(text, label, lang, source, category):
        text = str(text).strip()
        if 3 <= len(text) <= 512:
            rows.append({
                "text":     text,
                "label":    np.int8(label),
                "lang":     lang,
                "source":   source,
                "category": category,
            })

    # ── 1. synthetic_concat ───────────────────────────────────────────────────
    cat_lbl = ("concatenation", 1)
    n = targets.get(cat_lbl, 0)
    print(f"[augment] synthetic_concat: generating {n} rows")
    sample = random.choices(toxic_df.to_dict("records"), k=n)
    for rec in sample:
        text = aug_full_concat(rec["text"])
        add_row(text, 1, rec["lang"], "synthetic_concat", "concatenation")

    # ── 2. synthetic_partial_concat ───────────────────────────────────────────
    n_toxic = targets.get(("concatenation", 1), 0)
    n_clean = targets.get(("concatenation", 0), 0)
    print(f"[augment] synthetic_partial_concat: {n_toxic} toxic, {n_clean} clean")

    for rec in random.choices(toxic_df.to_dict("records"), k=n_toxic):
        text = aug_partial_concat(rec["text"], vocab=vocab)
        add_row(text, 1, rec["lang"], "synthetic_partial_concat", "concatenation")

    for rec in random.choices(clean_df.to_dict("records"), k=n_clean):
        text = aug_partial_concat(rec["text"], vocab=vocab)
        add_row(text, 0, rec["lang"], "synthetic_partial_concat", "concatenation")

    # ── 3. synthetic_context_filler ───────────────────────────────────────────
    cat_lbl = ("context", 1)
    n = targets.get(cat_lbl, 0)
    print(f"[augment] synthetic_context_filler: generating {n} rows")
    for rec in random.choices(toxic_df.to_dict("records"), k=n):
        text = aug_context_filler(rec["text"], filler_pool=filler_pool, n_fillers=random.randint(1, 2))
        add_row(text, 1, rec["lang"], "synthetic_context_filler", "context")

    # ── 4. synthetic_context_mix ──────────────────────────────────────────────
    n_mix = targets.get(("context", 0), 0) + targets.get(("context_ambiguous", 0), 0)
    print(f"[augment] synthetic_context_mix: generating ~{n_mix} rows")
    toxic_recs = random.choices(toxic_df.to_dict("records"), k=n_mix)
    clean_recs = random.choices(clean_df.to_dict("records"), k=n_mix)
    for t_rec, c_rec in zip(toxic_recs, clean_recs):
        text, label, category = _build_context_mix_row(t_rec["text"], c_rec["text"])
        add_row(text, label, t_rec["lang"], "synthetic_context_mix", category)

    # ── 5. synthetic_crosslang ────────────────────────────────────────────────
    n = targets.get(("concatenation", 1), 0) // 2  # share budget with concat
    print(f"[augment] synthetic_crosslang: generating {n} rows")
    for rec in random.choices(toxic_df.to_dict("records"), k=n):
        text = aug_crosslang_concat(rec["text"], lang=rec["lang"], lang_pool=lang_pool)
        if text:
            add_row(text, 1, "mixed", "synthetic_crosslang", "concatenation")

    synth_df = pd.DataFrame(rows)
    if synth_df.empty:
        print("[augment] Warning: no synthetic rows generated.")
        return synth_df

    synth_df["label"] = synth_df["label"].astype("int8")
    print(f"\n[augment] Synthetic rows generated: {len(synth_df):,}")
    print(synth_df.groupby(["source", "category", "label"]).size().to_string())
    return synth_df


# ── integration shim ──────────────────────────────────────────────────────────

def augment_and_combine(real_df: pd.DataFrame) -> pd.DataFrame:
    real_df = real_df.copy()
    real_df["label"] = (pd.to_numeric(real_df["label"], errors="coerce").fillna(0.0) >= 0.5).astype("int8")

    synth_df = build_augmented_dataset(real_df)

    combined = pd.concat([real_df, synth_df], ignore_index=True)
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len().between(3, 512)]
    combined = combined.dropna(subset=["text", "label"])
    combined["label"] = (pd.to_numeric(combined["label"], errors="coerce").fillna(0.0) >= 0.5).astype("int8")

    combined["_key"] = combined["text"].str.lower()
    combined = combined.drop_duplicates(subset="_key").drop(columns="_key")

    print(f"\n[augment] Final combined dataset: {len(combined):,} rows")
    print(f"  Real:      {len(real_df):,}")
    print(f"  Synthetic: {len(combined) - len(real_df):,}")
    print(f"  Synthetic ratio: {(len(combined) - len(real_df)) / len(combined):.1%}")
    return combined