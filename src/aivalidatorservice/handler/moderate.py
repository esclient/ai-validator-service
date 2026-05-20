from typing import Any

import grpc

from aivalidatorservice.grpc import moderation_pb2
from aivalidatorservice.logger.custom_logger import get_logger
from aivalidatorservice.service.service import ModerationService

log = get_logger(__name__)


async def moderate(
    service: ModerationService,
    request: moderation_pb2.ModerateObjectRequest,
    _context: grpc.aio.ServicerContext[Any, Any],
) -> moderation_pb2.ModerateObjectResponse:
    try:
        toxicity_level = await service.moderate(request.text)
    except Exception as exc:
        log.error(f"Moderation request failed: {exc}")
        raise

    response = moderation_pb2.ModerateObjectResponse(level=toxicity_level)
    log.debug(
        f"Responding ModerateObject id={request.id} toxicity_level={response.level}"
    )
    return response
