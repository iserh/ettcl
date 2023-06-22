from abc import ABC, abstractmethod
from dataclasses import dataclass

from ettcl.encoding.encoder import Encoder
from ettcl.utils.utils import Devices


@dataclass
class IndexerConfig:
    nbits: int = 2
    kmeans_niters: int = None


class BaseIndexer(ABC):
    def __init__(self, encoder: Encoder, config: IndexerConfig = IndexerConfig()) -> None:
        self.encoder = encoder
        self.config = config

    @abstractmethod
    def index(
        self,
        index_path: str,
        collection: list[str],
        gpus: Devices = 0,
        n_processes: int | None = None,
        resume: bool = False,
    ) -> None:
        pass
