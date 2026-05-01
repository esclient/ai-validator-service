import grpc

from aivalidatorservice.grpc import moderation_pb2
from aivalidatorservice.service.service import ModerationService
from logger.custom_logger import get_logger

log = get_logger(__name__)


async def moderate(
    service: ModerationService,
    request: moderation_pb2.ModerateObjectRequest,
    _context: grpc.ServicerContext,
) -> moderation_pb2.ModerateObjectResponse:
    try:
        is_toxic = await service.moderate(request.text)
    except Exception as exc:
        log.error(f"Moderation request failed: {exc}")
        raise

    response = moderation_pb2.ModerateObjectResponse(success=is_toxic)
    log.debug(
        f"Responding ModerateObject id={request.id} success={response.success}"
    )
    return response
