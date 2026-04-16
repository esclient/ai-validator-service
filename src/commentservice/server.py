import asyncio
import logging
from concurrent import futures

import asyncpg
import grpc
from grpc_reflection.v1alpha import reflection

from commentservice.clients.moderation_service import ModerationService
from commentservice.grpc import comment_pb2, comment_pb2_grpc
from commentservice.handler.handler import CommentHandler
from commentservice.repository.repository import CommentRepository
from commentservice.service.service import CommentService
from commentservice.settings import Settings


async def serve() -> None:
    settings = Settings()
    settings.configure_logging()
    logger = logging.getLogger(__name__)

    db_pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=1,
        max_size=10,
    )

    repo = CommentRepository(db_pool)
    loop = asyncio.get_running_loop()

    moderation_service = ModerationService(repo=repo, loop=loop)
    service = CommentService(repo, moderation_service)
    handler = CommentHandler(service)

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=5))
    comment_pb2_grpc.add_CommentServiceServicer_to_server(
        handler,
        server,
    )  # type: ignore[no-untyped-call]

    SERVICE_NAMES = (
        comment_pb2.DESCRIPTOR.services_by_name["CommentService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f"{settings.host}:{settings.port}")
    await server.start()
    logger.info(f"gRPC server listening on {settings.host}:{settings.port}")
    logger.info("Kafka consumer running in background")
    await server.wait_for_termination()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
