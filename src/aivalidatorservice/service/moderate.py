from aivalidatorservice.model.loader import ModerationModel

async def moderate(model: ModerationModel, normalized_text: str) -> bool:
    return model.predict(normalized_text)