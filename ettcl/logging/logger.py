import logging
import os

old_factory = logging.getLogRecordFactory()

def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.rank = os.environ.get("RANK", "#")
    return record

logging.setLogRecordFactory(record_factory)


def configure_logger(log_level: int | str) -> None:
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level)

    logging.basicConfig(
        format='[%(rank)s][%(asctime)s] [%(levelname)s] [%(funcName)s] %(message)s',
        level=log_level,
    )
