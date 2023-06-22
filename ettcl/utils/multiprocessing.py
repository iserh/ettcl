from typing import Callable
import os
import torch


def run_multiprocessed(function: Callable) -> Callable:

    def setup_process(*args, **kwargs):
        args, rank = args[:-1], args[-1]

        device = None
        if torch.cuda.is_available():
            device = rank % torch.cuda.device_count()

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        torch.cuda.device_count.cache_clear()

        with torch.cuda.device(device):
            return function(*args, rank=rank, **kwargs)

    return setup_process
