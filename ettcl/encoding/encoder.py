from abc import ABC, abstractmethod
from typing import Any, TypeVar

import torch
import wrapt

from ettcl.utils.multiprocessing import run_multiprocessed

TEncoder = TypeVar("TEncoder", bound="Encoder")


class Encoder(ABC):
    """`Interface` that provides methods to encode text to embedding vectors."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @abstractmethod
    def cuda(self: TEncoder, device: int | None = None) -> TEncoder:
        pass

    @abstractmethod
    def cpu(self: TEncoder) -> TEncoder:
        pass

    @abstractmethod
    def encode_passages(self, passages: list[str], **kwargs) -> Any:
        pass

    @abstractmethod
    def encode_queries(self, queries: list[str], **kwargs) -> Any:
        pass


class SingleVectorEncoder(Encoder):
    """`Interface` for single vector encoders (e.g. sentence-transformer)"""

    @abstractmethod
    def encode_passages(self, passages: list[str], **kwargs) -> torch.FloatTensor:
        pass

    @abstractmethod
    def encode_queries(self, queries: list[str], **kwargs) -> torch.FloatTensor:
        pass


class MultiVectorEncoder(Encoder):
    """`Interface` for multi vector encoders (e.g. colbert / late-interaction models)"""

    @abstractmethod
    def encode_passages(self, passages: list[str], **kwargs) -> tuple[torch.FloatTensor, list[int]]:
        pass

    @abstractmethod
    def encode_queries(self, queries: list[str], **kwargs) -> torch.FloatTensor:
        pass


class MultiProcessedEncoder(Encoder):
    """Wrapper that calls methods multiprocessed"""

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder

    def embedding_dim(self) -> int:
        return self.encoder.embedding_dim

    def cuda(self: TEncoder, device: int | None = None) -> TEncoder:
        self.encoder.cuda(device)
        return self

    def cpu(self: TEncoder) -> TEncoder:
        self.encoder.cpu()
        return self

    @run_multiprocessed
    def encode_passages(
        self, passages: list[str], *, rank: int, return_dict: bool = True, progress_bar: bool = False, **kwargs
    ) -> dict:
        if torch.cuda.is_available():
            self.encoder.cuda()

        return self.encoder.encode_passages(passages, return_dict=return_dict, progress_bar=progress_bar, **kwargs)

    @run_multiprocessed
    def encode_queries(
        self, passages: list[str], *, rank: int, return_dict: bool = True, progress_bar: bool = False, **kwargs
    ) -> dict:
        if torch.cuda.is_available():
            self.encoder.cuda()

        return self.encoder.encode_queries(passages, return_dict=return_dict, progress_bar=progress_bar, **kwargs)
