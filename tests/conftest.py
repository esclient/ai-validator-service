import pytest
import torch
from unittest.mock import MagicMock
from aivalidatorservice.model import loader
from aivalidatorservice.model.loader import ModerationModel
from aivalidatorservice.service.service import ModerationService

def make_fake_session(logit: float) -> MagicMock:
    session = MagicMock()
    outputs = MagicMock()
    outputs.logits = torch.tensor([[0.0, logit]], dtype=torch.float32)
    session.return_value = outputs
    return session

def make_fake_tokenizer(max_length: int = 128) -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.ones((1, max_length), dtype=torch.int64),
        "attention_mask": torch.ones((1, max_length), dtype=torch.int64),
    }
    return tokenizer

@pytest.fixture(scope="session")
def real_tokenizer():
    return loader.AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

@pytest.fixture
def toxic_service(monkeypatch):
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", lambda _: make_fake_tokenizer())
    monkeypatch.setattr(
        loader.ORTModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: make_fake_session(logit=5.0),
    )
    model = ModerationModel(model_path="mock-model", tokenizer_path="mock-tokenizer")
    return ModerationService(model)

@pytest.fixture
def clean_service(monkeypatch):
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", lambda _: make_fake_tokenizer())
    monkeypatch.setattr(
        loader.ORTModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: make_fake_session(logit=-5.0),
    )
    model = ModerationModel(model_path="mock-model", tokenizer_path="mock-tokenizer")
    return ModerationService(model)