from pathlib import Path

import torch
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from logger.custom_logger import get_logger

DATA_DIR = Path("data/processed")
OUT_DIR = Path("models/deberta-qat")
ONNX_DIR = Path("models/deberta-qat-onnx")
BEST_DIR = Path("models/deberta-qat-best")

for d in [OUT_DIR, ONNX_DIR, BEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BASE_ID = "microsoft/deberta-v3-small"
MODEL_ID = "esclient/deberta-toxicity-model"
log = get_logger(__name__)

log.info(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=2
)

main_export(
    model_name_or_path="esclient/deberta-toxicity-model-qat",
    output="models/deberta-qat-onnx-v2",
    task="text-classification",
    opset=14,
)

model = ORTModelForSequenceClassification.from_pretrained(
    "models/deberta-qat-onnx-v2"
)
inputs = tokenizer(
    "Kill yourself you worthless piece of garbage.",
    return_tensors="pt",
    max_length=128,
    truncation=True,
)
outputs = model(**inputs)
log.debug(f"ONNX logits softmax: {torch.softmax(outputs.logits, dim=-1)}")
