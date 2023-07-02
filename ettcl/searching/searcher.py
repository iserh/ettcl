from abc import ABC, abstractmethod
from collections import UserDict
from enum import Enum
from typing import Any

import numpy as np
import torch

from ettcl.indexing.indexer import IndexPath

TextQueries = str | list[str] | dict[int | str, str]
TensorLike = torch.Tensor | np.ndarray | list


class BatchResult(UserDict):
    def __init__(
        self,
        match_pids: TensorLike,
        match_scores: TensorLike,
    ):
        super().__init__(
            {
                "match_pids": match_pids,
                "match_scores": match_scores,
            }
        )

    def __getattr__(self, item: str) -> TensorLike:
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()


class TensorType(str, Enum):
    """
        Possible values for the `return_tensors` argument in [`ColBERTSearcher.search`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    NUMPY = "np"


class Searcher(ABC):
    """`Interface` that provides methods to search for most
    similar passage ids given query texts"""

    @property
    @abstractmethod
    def index_path(self) -> IndexPath:
        pass

    @index_path.setter
    @abstractmethod
    def index_path(self, path: str | IndexPath) -> None:
        pass

    @abstractmethod
    def search(
        self,
        queries: TextQueries,
        k: int = 10,
        *,
        return_tensors: bool | str | TensorType = "pt",
        use_gpu: bool = True,
        **kwargs,
    ) -> BatchResult:
        pass
