import logging


def configure_logger(log_level: int | str) -> None:
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level)
    logging.basicConfig(
        # format='%(asctime)s [%(levelname)s] "%(pathname)s", line %(lineno)d in %(funcName)s\t %(message)s',
        format='[%(asctime)s] [%(levelname)s] [%(funcName)s] %(message)s',
        level=log_level
    )
