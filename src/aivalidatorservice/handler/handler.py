from typing import Any

import grpc

from aivalidatorservice.grpc import moderation_pb2, moderation_pb2_grpc
from aivalidatorservice.handler.moderate import moderate as _moderate
from aivalidatorservice.logger.custom_logger import get_logger
from aivalidatorservice.service.service import ModerationService

log = get_logger(__name__)


class ModerationHandler(moderation_pb2_grpc.ModerationServiceServicer):
    def __init__(self, service: ModerationService):
        self._service = service

    async def ModerateObject(
        self,
        request: moderation_pb2.ModerateObjectRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> moderation_pb2.ModerateObjectResponse:
        log.debug(
            f"Received ModerateObject request id={request.id} type={request.type}"
        )
        return await _moderate(self._service, request, context)
