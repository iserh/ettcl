from abc import ABC, abstractmethod

import torch


class Encoder(ABC):
    """`Interface` that provides methods to encode text to embedding vectors."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @abstractmethod
    def cuda(self, device: int | None = None) -> "Encoder":
        pass

    @abstractmethod
    def cpu(self) -> "Encoder":
        pass

    @abstractmethod
    def encode_passages(self, passages: list[str]) -> tuple[torch.FloatTensor, list[int]]:
        pass

    @abstractmethod
    def encode_queries(self, queries: list[str]) -> torch.FloatTensor:
        pass
