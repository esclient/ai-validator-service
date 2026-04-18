import os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

DATA_DIR = Path("data/processed")
OUT_DIR  = Path("models/deberta-toxicity")
MODEL_ID = "microsoft/deberta-v3-small"
MAX_LEN  = 64


@dataclass
class Config:
    batch_size:      int   = 64
    grad_accum:      int   = 2
    epochs:          int   = 3
    lr:              float = 2e-5
    warmup_ratio:    float = 0.1
    weight_decay:    float = 0.01
    fp16:            bool  = torch.cuda.is_available()
    dataloader_workers: int = 4


cfg = Config()


class ToxicityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.encodings = tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get("token_type_ids", {idx: None})[idx]
                              if "token_type_ids" in self.encodings else None,
            "labels":         self.labels[idx],
        }

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
    return {
        "f1":        f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "roc_auc":   roc_auc_score(labels, probs),
    }


def per_lang_metrics(trainer, dataset_df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    rows = []
    for lang, grp in dataset_df.groupby("lang"):
        if len(grp) < 50:
            continue
        ds = ToxicityDataset(grp.reset_index(drop=True), tokenizer, MAX_LEN)
        out = trainer.predict(ds)
        preds = np.argmax(out.predictions, axis=-1)
        rows.append({
            "lang":      lang,
            "n":         len(grp),
            "f1":        f1_score(grp["label"].values, preds, zero_division=0),
            "recall":    recall_score(grp["label"].values, preds, zero_division=0),
            "precision": precision_score(grp["label"].values, preds, zero_division=0),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def main():
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df   = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df  = pd.read_parquet(DATA_DIR / "test.parquet")

    n_clean = (train_df["label"] == 0).sum()
    n_toxic = (train_df["label"] == 1).sum()
    total   = len(train_df)
    class_weights = torch.tensor(
        [total / (2 * n_clean), total / (2 * n_toxic)], dtype=torch.float32
    )
    print(f"Class weights  clean: {class_weights[0]:.3f}, toxic: {class_weights[1]:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

    train_ds = ToxicityDataset(train_df, tokenizer, MAX_LEN)
    val_ds   = ToxicityDataset(val_df,   tokenizer, MAX_LEN)
    test_ds  = ToxicityDataset(test_df,  tokenizer, MAX_LEN)

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        dataloader_num_workers=cfg.dataloader_workers,
        report_to="none",
        logging_steps=100,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    print("\n=== Test set ===")
    test_results = trainer.predict(test_ds)
    preds = np.argmax(test_results.predictions, axis=-1)
    probs = torch.softmax(
        torch.tensor(test_results.predictions, dtype=torch.float32), dim=-1
    )[:, 1].numpy()
    print({
        "f1":        f1_score(test_df["label"].values, preds),
        "precision": precision_score(test_df["label"].values, preds),
        "recall":    recall_score(test_df["label"].values, preds),
        "roc_auc":   roc_auc_score(test_df["label"].values, probs),
    })

    if "lang" in test_df.columns:
        print("\n=== Per-language (test) ===")
        lang_df = per_lang_metrics(trainer, test_df, tokenizer)
        print(lang_df.to_string(index=False))
        lang_df.to_csv(OUT_DIR / "per_lang_metrics.csv", index=False)

    trainer.save_model(str(OUT_DIR / "best"))
    tokenizer.save_pretrained(str(OUT_DIR / "best"))
    print(f"\nModel saved to {OUT_DIR / 'best'}")


if __name__ == "__main__":
    main()