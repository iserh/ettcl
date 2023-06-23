from time import perf_counter

import torch

Devices = int | bool | list[int] | list[str] | None


def to_gpu_list(gpus: Devices) -> list[int]:
    """List of selected gpu devices."""
    # convert to list of selected gpus
    match gpus:
        case bool():
            gpus = list(range(torch.cuda.device_count())) if gpus else []
        case int():
            gpus = list(range(gpus))
        case str():
            gpus = gpus.split(",")
        case None:
            gpus = []

    # cast list items to integer
    gpus = list(map(int, gpus))
    # sort list
    gpus = sorted(list(set(gpus)))

    # make sure only visible devices are selected
    assert all(
        device_idx in range(torch.cuda.device_count()) for device_idx in gpus
    ), f"Not all devices from {gpus} are visible!"

    return gpus


class catchtime:
    def __init__(self, output_to_console: bool = True, desc: str = "Time"):
        self.output_to_console = output_to_console
        self.desc = desc

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        if self.output_to_console:
            self.readout = f"{self.desc}: {self.time:.3f} seconds"
            print(self.readout)
