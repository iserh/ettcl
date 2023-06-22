from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from torch import Tensor

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


class Searcher(ABC):
    """`Interface` that provides methods to search for most
    similar passage ids given query texts"""

    @abstractmethod
    def search(
        self,
        queries: TextQueries,
        k: int,
        return_tensors: bool | str | TensorType = "pt",
        gpu: bool | int = True,
        progress_bar: bool = True,
    ) -> SearchResult:
        pass
