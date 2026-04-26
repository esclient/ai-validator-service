import grpc
from aivalidatorservice.grpc import moderation_pb2
from aivalidatorservice.service.service import ModerationService

async def Moderate(
    service: ModerationService,
    request: moderation_pb2.ModerateObjectRequest,
    context: grpc.ServicerContext,
) -> moderation_pb2.ModerateObjectResponse:
    is_toxic = await service.moderate(request.text)
    return moderation_pb2.ModerateObjectResponse(success=is_toxic)