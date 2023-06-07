from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from ettcl.core.config import SetupConfig
from ettcl.encoding.base_encoder import EncoderFactory


@dataclass
class IndexingArguments(SetupConfig):
    nranks: int = 1
    nbits: int = 2
    dim: int = 128


class BaseIndexer(ABC):
    def __init__(
        self, encoder_factory: EncoderFactory, args: IndexingArguments = IndexingArguments()
    ) -> None:
        self.encoder_factory = encoder_factory
        self.args = args

    @abstractmethod
    def index(self, index_path: str, collection: list[str], resume: bool = False) -> None:
        pass
