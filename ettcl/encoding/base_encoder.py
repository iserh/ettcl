from abc import ABC, abstractmethod

import torch


class BaseEncoder(ABC):
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @abstractmethod
    def cuda(self) -> "BaseEncoder":
        pass

    @abstractmethod
    def cpu(self) -> "BaseEncoder":
        pass

    @abstractmethod
    def encode_passages(self, passages: list[str]) -> tuple[torch.FloatTensor, list[int]]:
        pass

    @abstractmethod
    def encode_queries(self, queries: list[str]) -> torch.FloatTensor:
        pass
