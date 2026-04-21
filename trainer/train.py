# ── This code is optimized for Google Colab ──────────────────────────────
import os, numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import shutil

# Confirm GPU + precision
print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

DATA_DIR = Path("data/processed")
OUT_DIR  = Path("models/deberta-toxicity")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ID = "microsoft/deberta-v3-small"
MAX_LEN  = 128

class SaveBestToDriveCallback(TrainerCallback):
    def __init__(self, drive_path, tokenizer, f1_weight=0.5, auc_weight=0.5):
        self.drive_path = drive_path
        self.tokenizer = tokenizer
        self.f1_weight = f1_weight
        self.auc_weight = auc_weight
        self.best_score = 0.0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_f1  = metrics.get("eval_f1", 0.0)
        current_auc = metrics.get("eval_roc_auc", 0.0)
        
        combined = (self.f1_weight * current_f1) + (self.auc_weight * current_auc)
        
        print(f"F1: {current_f1:.4f} | AUC: {current_auc:.4f} | Combined: {combined:.4f} (best: {self.best_score:.4f})")
        
        if combined > self.best_score:
            self.best_score = combined
            print(f"✓ New best combined score: {combined:.4f} - saving to Drive...")
            
            if os.path.exists(self.drive_path):
                shutil.rmtree(self.drive_path)
            
            kwargs["model"].save_pretrained(self.drive_path)
            self.tokenizer.save_pretrained(self.drive_path)
            print(f"✓ Saved to {self.drive_path}")

class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, name="dataset"):
        print(f"Tokenizing {name} ({len(df)} samples)...")
        enc = tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
            verbose=False,
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = torch.tensor(df["label"].values, dtype=torch.long)
        print(f"Done tokenizing {name}!")

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits shape is (n, 2) — use argmax for preds, softmax[:,1] for probs
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(
        torch.tensor(logits, dtype=torch.float32), dim=-1
    )[:, 1].numpy()
    return {
        "f1":        f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "roc_auc":   roc_auc_score(labels, probs),
    }

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # tensor([w_clean, w_toxic])

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")          # long, shape (B,)
        outputs = model(**inputs)
        logits = outputs.logits                # shape (B, 2) — keep both classes

        loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=self.class_weights.to(logits.device),
        )
        return (loss, outputs) if return_outputs else loss


# ── Data ─────────────────────────────────────────────────────────────────────
train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df   = pd.read_parquet(DATA_DIR / "val.parquet")
test_df  = pd.read_parquet(DATA_DIR / "test.parquet")

def stratified_sample(df, n, label_col="label", seed=42):
    return (
        df.groupby(label_col, group_keys=False)
          .apply(lambda x: x.sample(min(len(x), int(n * len(x) / len(df))), random_state=seed))
          .sample(frac=1, random_state=seed)  # shuffle
          .reset_index(drop=True)
    )

def language_balanced_sample(df, en_cap=300_000, ru_cap=300_000, other_per_lang=8_000, seed=42):
    """
    Cap EN and RU, keep a small slice of each other language for context.
    Preserves label balance within each language slice.
    """
    parts = []
    
    for lang, group in df.groupby("lang"):
        if lang == "en":
            cap = en_cap
        elif lang == "ru":
            cap = ru_cap
        elif lang == "mixed":
            cap = 20_000  
        else:
            cap = other_per_lang
        
        cap_n = min(len(group), cap)
        # stratify within the language group
        sampled = (
            group.groupby("label", group_keys=False)
                 .apply(lambda x: x.sample(min(len(x), int(cap_n * len(x) / len(group))), random_state=seed))
                 .sample(frac=1, random_state=seed)
                 .reset_index(drop=True)
        )
        parts.append(sampled)
    
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

train_df = language_balanced_sample(train_df, en_cap=300_000, ru_cap=300_000, other_per_lang=8_000)

train_df = stratified_sample(train_df, n=150_000)
val_df   = stratified_sample(val_df,   n=20_000)

print(f"Train: {len(train_df):,} rows")
print(f"Val:   {len(val_df):,} rows")
print(f"Test:  {len(test_df):,} rows")
print(f"Avg text length: {train_df['text'].str.len().mean():.0f} chars")

n_clean, n_toxic = (train_df["label"]==0).sum(), (train_df["label"]==1).sum()
class_weights = torch.tensor(
    [np.sqrt(len(train_df)/(2*n_clean)), np.sqrt(len(train_df)/(2*n_toxic))],
    dtype=torch.float32
)
print(f"Class weights  clean: {class_weights[0]:.3f}, toxic: {class_weights[1]:.3f}")

print(f"n_clean: {n_clean:,}, n_toxic: {n_toxic:,}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

train_ds = ToxicityDataset(train_df, tokenizer, MAX_LEN, name="train")
val_ds   = ToxicityDataset(val_df,   tokenizer, MAX_LEN, name="val")

# ── Training args — fp16/bf16 hardcoded, no cfg object ───────────────────────
args = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=4,
    learning_rate=8e-7,          # lowered from 2e-5
    warmup_ratio=0.06,
    weight_decay=0.01,
    max_grad_norm=1.0,           # explicit gradient clipping
    fp16=False,
    bf16=True,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=100,              # evaluate more frequently to catch issues early
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    dataloader_num_workers=2,
    report_to="none",
    logging_steps=50,            # log more frequently
)

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5),
        SaveBestToDriveCallback(
            drive_path="/content/drive/MyDrive/deberta-toxicity-best",
            tokenizer=tokenizer,
        ),
    ],
)

model.eval()
with torch.no_grad():
    sample = {k: v[:8].to(model.device) for k, v in {
        "input_ids": train_ds.input_ids,
        "attention_mask": train_ds.attention_mask
    }.items()}
    out = model(**sample)
    print("Logits:", out.logits)
    print("Preds:", out.logits.argmax(-1))
    print("Labels:", train_ds.labels[:8])

trainer.train()

test_ds = ToxicityDataset(test_df, tokenizer, MAX_LEN, name="test")
test_results = trainer.evaluate(test_ds)
print(test_results)