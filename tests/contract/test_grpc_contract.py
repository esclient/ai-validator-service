import pytest

from aivalidatorservice.grpc import moderation_pb2
from aivalidatorservice.handler.handler import ModerationHandler


@pytest.mark.asyncio
async def test_grpc_contract_toxic_input_returns_expected_response(
    toxic_service,
):
    request = moderation_pb2.ModerateObjectRequest(
        id=123,
        type=moderation_pb2.OBJECT_TYPE_COMMENT_TEXT,
        text="Ihateyou",
    )
    handler = ModerationHandler(toxic_service)
    response = await handler.ModerateObject(request, context=None)
    assert isinstance(response, moderation_pb2.ModerateObjectResponse)
    assert response.success is True
