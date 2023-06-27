import os
from typing import Callable

import multiprocess as mp
import torch
from ettcl.logging.logger import configure_logger
import logging

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass


def run_multiprocessed(function: Callable) -> Callable:
    def setup_process(*args, **kwargs):
        args, rank = args[:-1], args[-1]

        os.environ["RANK"] = str(rank)
        configure_logger(logging.WARNING)
        logger = logging.getLogger("colbert")
        logger.setLevel(logging.WARNING)

        device = None
        if torch.cuda.is_available():
            device = rank % torch.cuda.device_count()

        with torch.cuda.device(device):
            return function(*args, rank=rank, **kwargs)

    return setup_process
