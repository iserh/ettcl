import logging
import os
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


def configure_logger(log_level: int | str) -> None:
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = os.environ.get("RANK", "#")
        return record

    logging.setLogRecordFactory(record_factory)

    if isinstance(log_level, str):
        log_level = getattr(logging, log_level)

    logging.basicConfig(
        format="[%(rank)s][%(asctime)s] [%(levelname)s] [%(funcName)s] %(message)s",
        level=log_level,
    )


@contextmanager
def memory_stats() -> None:
    for idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(idx)

    try:
        yield

    finally:
        for idx in range(torch.cuda.device_count()):
            try:
                logger.debug("\n" + torch.cuda.memory_summary(idx, True))
            except KeyError:
                logger.debug(f"Could not print memory stats for device {idx}")


def profile_memory(fn: callable) -> callable:
    def fn_wrapper(*args, **kwargs):
        with memory_stats():
            return fn(*args, **kwargs)

    return fn_wrapper
