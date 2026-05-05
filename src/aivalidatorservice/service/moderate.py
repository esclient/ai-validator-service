from aivalidatorservice.logger.custom_logger import get_logger
from aivalidatorservice.model.loader import ModerationModel

log = get_logger(__name__)


def moderate(model: ModerationModel, normalized_text: str) -> bool:
    log.debug("Running model prediction in service.moderate")
    return model.predict(normalized_text)
