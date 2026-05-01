import asyncio

from aivalidatorservice.model.loader import ModerationModel
from aivalidatorservice.service.moderate import moderate as _moderate
from custom_logger import get_logger

log = get_logger(__name__)


class ModerationService:
    def __init__(self, model: ModerationModel):
        self._model = model

    async def moderate(self, normalized_text: str) -> bool:
        log.debug(f"Service moderating text length={len(normalized_text)}")
        result = await asyncio.to_thread(
            _moderate, self._model, normalized_text
        )
        log.debug(f"Service moderation result={result}")
        return result
