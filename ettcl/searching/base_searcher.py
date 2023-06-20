from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from torch import Tensor

from ettcl.encoding.base_encoder import BaseEncoder
from enum import Enum

TextQueries = str | list[str] | dict[int | str, str]
T = list | np.ndarray | Tensor
SearchResult = tuple[T, T] | tuple[list[T], list[T]]


class TensorType(str, Enum):
    """
        Possible values for the `return_tensors` argument in [`ColBERTSearcher.search`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    NUMPY = "np"


@dataclass
class SearcherConfig:
    ncells: int | None = None


class BaseSearcher(ABC):
    def __init__(
        self, index_path: str, encoder: BaseEncoder, args: SearcherConfig = SearcherConfig()
    ) -> None:
        self.index_path = index_path
        self.encoder = encoder
        self.args = args

    @abstractmethod
    def search(
        self,
        queries: TextQueries,
        k: int,
        return_tensors: bool | str | TensorType = "pt",
        gpu: bool | int = True,
        progress_bar: bool = False,
    ) -> SearchResult:
        pass
