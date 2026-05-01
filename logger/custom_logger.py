import datetime
import logging


class AbseilStyleFormatter(logging.Formatter):
    def format(self, record):
        level = record.levelname
        location = f"{record.filename}:{record.lineno}"
        timespan = (
            datetime.datetime.now(datetime.UTC).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3]
            + "Z"
        )

        subsystem = getattr(record, "subsystem", "app")
        code = getattr(record, "code", "")
        explanation = getattr(record, "explanation", "")

        msg = f"[{level}] {location} timespan={timespan} subsystem={subsystem}"
        if code:
            msg += f" code={code}"
        msg += f' message="{record.getMessage()}"'
        if explanation:
            msg += f' explanation="{explanation}"'

        return msg


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(AbseilStyleFormatter())
        logger.addHandler(handler)

    return logger
