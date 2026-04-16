import asyncio
import logging
from concurrent.futures import Future

from commentservice.grpc import moderation_pb2
from commentservice.repository.repository import CommentRepository

from .kafka.config import KafkaConfig
from .kafka.consumer import ModerationResponseConsumer
from .kafka.producer import ModerationRequestProducer

logger = logging.getLogger(__name__)


class ModerationService:
    def __init__(
        self, repo: CommentRepository, loop: asyncio.AbstractEventLoop
    ):
        self.config = KafkaConfig()
        self._repo = repo
        self._loop = loop

        self.producer = ModerationRequestProducer(self.config)
        self.consumer = ModerationResponseConsumer(
            self.config, callback=self._handle_moderation_response
        )

        self.consumer.start()

    def request_moderaiton(self, comment_id: int, comment_text: str) -> bool:
        try:
            request = moderation_pb2.ModerateObjectRequest(
                id=comment_id,
                text=comment_text,
                type=moderation_pb2.OBJECT_TYPE_COMMENT_TEXT,
            )

            success = self.producer.send_moderation_request(request)

            if not success:
                logger.error(
                    f"send moderation request error: comment_id={comment_id}"
                )

            return success

        except Exception as e:
            logger.error(f"Error requesting moderation: {e}")
            return False

    def _handle_moderation_response(
        self, response: moderation_pb2.ModerateObjectResponse, request_id: int
    ) -> None:

        try:
            is_flagged = response.success
            self._update_comment_status(request_id, is_flagged)

        except Exception as e:
            logger.error(f"Error handling moderation response: {e}")

    def _update_comment_status(
        self, comment_id: int, is_flagged: bool
    ) -> None:
        coroutine = self._repo.update_comment_status(comment_id, is_flagged)

        future: Future[bool] = asyncio.run_coroutine_threadsafe(
            coroutine, self._loop
        )

        def _log_result(done_future: Future[bool]) -> None:
            try:
                success = done_future.result()
                if not success:
                    logger.error(
                        f"update comment status error: comment_id={comment_id}"
                    )
            except Exception as e:
                logger.error(f"Error updating comment status: {e}")

        future.add_done_callback(_log_result)

    def shutdown(self) -> None:
        self.consumer.stop()
        self.producer.flush()
