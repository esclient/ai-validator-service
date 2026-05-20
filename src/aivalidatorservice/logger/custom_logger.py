import datetime
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aivalidatorservice.settings import Settings

_configured_level = logging.INFO


class AbseilStyleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        location = f"{record.filename}:{record.lineno}"
        timespan = (
            datetime.datetime.now(datetime.UTC).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
            + "Z"
        )

        subsystem = getattr(record, "subsystem", "app")
        code = getattr(record, "code", "")
        explanation = getattr(record, "explanation", "")

        msg = f"[{level}] {location} timespan={timespan} subsystem={subsystem}"
        if code:
            msg += f" code={code}"
        msg += f" message={record.getMessage()}"
        if explanation:
            msg += f" explanation={explanation}"

        return msg


def configure_logger(settings: "Settings") -> None:
    global _configured_level
    _configured_level = getattr(
        logging, settings.log_level.upper(), logging.INFO
    )
    for _name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            logger.setLevel(_configured_level)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(_configured_level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(AbseilStyleFormatter())
        logger.addHandler(handler)

    return logger
