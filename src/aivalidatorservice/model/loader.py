import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from aivalidatorservice.grpc import moderation_pb2
from aivalidatorservice.logger.custom_logger import get_logger

TOXICITY_THRESHOLD_MODERATE = 0.5
TOXICITY_THRESHOLD_SEVERE = 0.85

log = get_logger(__name__)
__all__ = [
    "AutoTokenizer",
    "ModerationModel",
    "ORTModelForSequenceClassification",
]


class ModerationModel:
    def __init__(self, model_path: str, tokenizer_path: str):
        log.info(
            f"Initializing ModerationModel model_path={model_path} tokenizer_path={tokenizer_path}"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # type: ignore[no-untyped-call]
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

    def predict(self, text: str) -> moderation_pb2.ToxicityLevel:
        score = self._run(text)
        log.debug(f"Prediction toxic_score={score:.4f}")
        if score < TOXICITY_THRESHOLD_MODERATE:
            log.debug("Text toxicity classified as non-toxic")
            return moderation_pb2.TOXICITY_LEVEL_NONE
        elif score < TOXICITY_THRESHOLD_SEVERE:
            log.debug("Text toxicity classified as moderate")
            return moderation_pb2.TOXICITY_LEVEL_MODERATE
        else:
            log.debug("Text toxicity classified as severe")
            return moderation_pb2.TOXICITY_LEVEL_SEVERE
