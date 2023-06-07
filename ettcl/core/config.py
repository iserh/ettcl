from dataclasses import dataclass
from functools import cached_property

import torch


@dataclass
class SetupConfig:
    gpus: list[str | int] | str | int = 0

    @cached_property
    def gpus_(self) -> list[int]:
        """List of selected gpu devices."""
        gpus = self.gpus
        # convert to list of selected gpus
        match gpus:
            case int():
                gpus = list(range(gpus))
            case str():
                gpus = gpus.split(",")

        # cast list items to integer
        gpus = list(map(int, gpus))
        # sort list
        gpus = sorted(list(set(gpus)))

        # make sure only visible devices are selected
        assert all(
            device_idx in range(torch.cuda.device_count()) for device_idx in gpus
        ), f"Not all devices from {gpus} are visible!"

        return gpus
