from aivalidatorservice.service.moderate import moderate as _moderate
from aivalidatorservice.model.loader import ModerationModel

class ModerationService:
    def __init__(self, model: ModerationModel):
        self._model = model

    async def moderate(self, normalized_text: str) -> bool:
        return await _moderate(self._model, normalized_text)