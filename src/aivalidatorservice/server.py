import asyncio
import logging
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection

from aivalidatorservice.grpc import moderation_pb2, moderation_pb2_grpc
from aivalidatorservice.handler.handler import ModerationHandler
from aivalidatorservice.service.service import ModerationService
from aivalidatorservice.settings import Settings
from aivalidatorservice.model.loader import ModerationModel

async def serve() -> None:
    settings = Settings()
    settings.configure_logging()
    logger = logging.getLogger(__name__)

    model = ModerationModel(
    model_path="./deberta-int8",
    tokenizer_path="./deberta-int8",)

    service = ModerationService(model)
    handler = ModerationHandler(service)

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=5))
    moderation_pb2_grpc.add_ModerationServiceServicer_to_server(handler, server) # type: ignore[no-untyped-call]

    SERVICE_NAMES = (
        moderation_pb2.DESCRIPTOR.services_by_name["ModerationService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f"{settings.host}:{settings.port}")
    await server.start()
    logger.info(f"gRPC server listening on {settings.host}:{settings.port}")
    await server.wait_for_termination()

def main() -> None:
    asyncio.run(serve())

if __name__ == "__main__":
    main()