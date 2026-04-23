import grpc
from aivalidatorservice.grpc import moderation_pb2, moderation_pb2_grpc
from aivalidatorservice.handler.moderate import Moderate as _moderate
from aivalidatorservice.service.service import ModerationService

class ModerationHandler(moderation_pb2_grpc.ModerationServiceServicer):
    def __init__(self, service: ModerationService):
        self._service = service

    async def ModerateObject(
        self,
        request: moderation_pb2.ModerateObjectRequest,
        context: grpc.ServicerContext,
    ) -> moderation_pb2.ModerateObjectResponse:
        return await _moderate(self._service, request, context)