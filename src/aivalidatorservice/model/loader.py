import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from logger.custom_logger import get_logger

TOXICITY_THRESHOLD = 0.75

log = get_logger(__name__)


class ModerationModel:
    def __init__(self, model_path: str, tokenizer_path: str):
        log.info(
            f"Initializing ModerationModel model_path={model_path} tokenizer_path={tokenizer_path}"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._model = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            file_name="model_quantized.onnx",
        )
        self._run("warmup")

    def _run(self, text: str) -> float:
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        )
        outputs = self._model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        toxic_score = probabilities[0][1].item()
        return toxic_score

    def predict(self, text: str) -> bool:
        score = self._run(text)
        log.debug(
            f"Prediction toxic_score={score:.4f} threshold={TOXICITY_THRESHOLD}"
        )
        return score > TOXICITY_THRESHOLD
