from abc import ABC, abstractmethod
from pathlib import Path

from checksumdir import dirhash

from ettcl.utils.utils import Devices


class IndexPath(str):
    def __new__(cls, path: str):
        return super().__new__(cls, path)

    def __init__(self, path: str):
        path = Path(path)
        assert path.is_dir(), f"Index {path} does not exist."
        self.chksum = dirhash(path)


class Indexer(ABC):
    @abstractmethod
    def index(
        self,
        output_path: str | Path,
        collection: list[str],
        gpus: Devices = True,
        n_processes: int = -1,
        resume: bool = False,
    ) -> IndexPath:
        pass
