from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

from ettcl.core.config import SetupConfig


class BaseEncoder(ABC):
    def __init__(self, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu

        if self.use_gpu and torch.cuda.device_count() == 0:
            print(
                f"[WARNING] init argument `use_gpu` set to True in {self.__class__}, but cuda not available. Disabling"
            )
            self.use_gpu = False

    @abstractmethod
    def encode_passages(self, passages: list[str]) -> tuple[torch.FloatTensor, list[int]]:
        pass

    @abstractmethod
    def encode_queries(self, queries: list[str]) -> torch.FloatTensor:
        pass


Encoder = TypeVar("Encoder", bound=BaseEncoder)


class EncoderFactory:
    def __init__(self, EncoderCls: type[Encoder], **kwargs):
        self.EncoderCls = EncoderCls
        self.kwargs = kwargs

    def create(self, **overrides) -> BaseEncoder:
        kwargs = self.kwargs
        kwargs.update(overrides)
        return self.EncoderCls(**kwargs)
