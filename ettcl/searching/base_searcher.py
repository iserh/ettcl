from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import torch

from ettcl.core.config import SetupConfig
from ettcl.encoding.base_encoder import EncoderFactory

TextQueries = str | list[str] | dict[int | str | str]
SearchResult = (
    list[tuple[torch.LongTensor, torch.FloatTensor]] | tuple[torch.LongTensor, torch.FloatTensor]
)


@dataclass
class SearchingArguments(SetupConfig):
    ncells: int = 16


class BaseSearcher(ABC):
    def __init__(
        self,
        index_path: str,
        encoder_factory: EncoderFactory,
        args: SearchingArguments = SearchingArguments(),
    ) -> None:
        self.index_path = index_path
        self.encoder_factory = encoder_factory
        self.args = args

    @abstractmethod
    def search(self, queries: TextQueries, k: int) -> SearchResult:
        pass
