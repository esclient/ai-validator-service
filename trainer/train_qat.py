import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from logger.custom_logger import get_logger

log = get_logger(__name__)

log.info(f"PyTorch: {torch.__version__}")
log.info(f"CUDA: {torch.cuda.is_available()}")
device_name = (
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
)
log.info(f"Device: {device_name}")
if torch.cuda.is_available():
    vram_gb = round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
    )
    log.info(f"VRAM: {vram_gb} GB")

DATA_DIR = Path("data/processed")
OUT_DIR = Path("models/deberta-qat")
BEST_DIR = Path("models/deberta-qat-best")

for d in [OUT_DIR, BEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BASE_ID = "microsoft/deberta-v3-small"
MODEL_ID = "esclient/deberta-toxicity-model"
MAX_LEN = 128


class SaveBestCallback(TrainerCallback):
    def __init__(self, save_path, tokenizer, f1_weight=0.5, auc_weight=0.5):
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.f1_weight = f1_weight
        self.auc_weight = auc_weight
        self.best_score = 0.0

    def on_evaluate(self, _args, _state, _control, metrics=None, **kwargs):
        current_f1 = metrics.get("eval_f1", 0.0)
        current_auc = metrics.get("eval_roc_auc", 0.0)
        combined = (self.f1_weight * current_f1) + (
            self.auc_weight * current_auc
        )
        log.info(
            f"F1: {current_f1:.4f} | AUC: {current_auc:.4f} | Combined: {combined:.4f} (best: {self.best_score:.4f})"
        )
        if combined > self.best_score:
            self.best_score = combined
            if self.save_path.exists():
                shutil.rmtree(self.save_path)
            kwargs["model"].save_pretrained(self.save_path)
            self.tokenizer.save_pretrained(self.save_path)
            log.info(f"New best {combined:.4f} saved -> {self.save_path}")


class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, name="dataset"):
        log.info(f"Tokenizing {name} ({len(df):,} samples)...")
        enc = tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        log.info(f"Done tokenizing {name}!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[
        :, 1
    ].numpy()
    return {
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


log.info("Loading parquet files...")
train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df = pd.read_parquet(DATA_DIR / "val.parquet")


def stratified_sample(df, n, label_col="label", seed=42):
    return (
        df.groupby(label_col, group_keys=False)
        .apply(
            lambda x: x.sample(
                min(len(x), int(n * len(x) / len(df))), random_state=seed
            ),
            include_groups=True,
        )
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def language_balanced_sample(
    df, en_cap=300_000, ru_cap=300_000, other_per_lang=8_000, seed=42
):
    parts = []
    for lang, group in df.groupby("lang"):
        cap = {"en": en_cap, "ru": ru_cap, "mixed": 20_000}.get(
            lang, other_per_lang
        )
        cap_n = min(len(group), cap)
        parts.append(
            group.groupby("label", group_keys=False)
            .apply(
                lambda x, cap_n=cap_n, group_len=len(group): x.sample(
                    min(len(x), int(cap_n * len(x) / group_len)),
                    random_state=seed,
                ),
                include_groups=True,
            )
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
    return (
        pd.concat(parts)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


train_df = language_balanced_sample(train_df)
train_df = stratified_sample(train_df, n=50_000)
val_df = stratified_sample(val_df, n=10_000)

log.info(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

n_clean = (train_df["label"] == 0).sum()
n_toxic = (train_df["label"] == 1).sum()
class_weights = torch.tensor(
    [
        np.sqrt(len(train_df) / (2 * n_clean)),
        np.sqrt(len(train_df) / (2 * n_toxic)),
    ],
    dtype=torch.float32,
)
log.info(
    f"Weights — clean: {class_weights[0]:.3f}, toxic: {class_weights[1]:.3f}"
)

log.info(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=2
)

train_ds = ToxicityDataset(train_df, tokenizer, MAX_LEN, "train")
val_ds = ToxicityDataset(val_df, tokenizer, MAX_LEN, "val")

model.eval()
with torch.no_grad():
    sample = {
        "input_ids": train_ds.input_ids[:8],
        "attention_mask": train_ds.attention_mask[:8],
    }
    out = model(**sample)
    log.debug(f"Logits: {out.logits}")
    log.debug(f"Preds: {out.logits.argmax(-1)}")

# ── Training args ───────────────────────────────────
# This snippet of code has to be adjusted based on your GPU's VRAM.
# The current settings are for a 4GB card, which is quite tight for DeBERTa training.
# If you have more VRAM, you can increase batch sizes and reduce gradient accumulation steps for faster training.
# Also note that some GPUs support fp16 but not bf16, so adjust those settings accordingly.

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=8,
    learning_rate=2e-7,
    warmup_ratio=0.06,
    weight_decay=0.01,
    max_grad_norm=1.0,
    fp16=True,
    bf16=False,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=300,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    dataloader_num_workers=2,
    report_to="none",
    logging_steps=50,
)

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        SaveBestCallback(save_path=BEST_DIR, tokenizer=tokenizer),
    ],
)

trainer.train()
log.info("Training complete.")
