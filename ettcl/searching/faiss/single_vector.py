from dataclasses import dataclass
from logging import getLogger

import torch
from colbert.infra.config import BaseConfig, RunConfig, RunSettings, SearchSettings
from colbert.infra.run import Run
from colbert.search.index_storage import IndexScorer
from colbert.utils.utils import zipstar
from datasets import Dataset

from ettcl.encoding.encoder import Encoder
from ettcl.indexing.indexer import IndexPath
from ettcl.logging.tqdm import trange
from ettcl.searching.searcher import BatchResult, Searcher, TensorLike, TensorType, TextQueries
import json
from pathlib import Path


logger = getLogger(__name__)


class FaissSingleVectorSearcher(Searcher):
    def __init__(
        self,
        index_path: str | IndexPath,
        encoder: Encoder,
    ) -> None:
        self.encoder = encoder
        self.index_path = index_path

    @property
    def index_path(self) -> Path:
        return Path(self.__index_path)

    @index_path.setter
    def index_path(self, path: str | IndexPath) -> None:
        if path is not None:
            self.__index_path = IndexPath(path)

            with open(Path(path) / "metadata.json", "r") as fp:
                metadata = json.load(fp)

            self.ntotal = metadata["ntotal"]

        else:
            self.__index_path = None

    def search(
        self,
        queries: TextQueries,
        k: int = 10,
        *,
        return_tensors: bool | str | TensorType = False,
        use_gpu: bool = True,
        progress_bar: bool = True,
        rank: int | str = "#",
        **unused_kwargs,
    ) -> BatchResult:
        match queries:
            case str():
                queries = [queries]
            case dict():
                queries = queries.values()

        # disable progress bar in multiprocess
        progress_bar = progress_bar if rank == "#" else False

        use_gpu = use_gpu and torch.cuda.is_available()
        device = torch.cuda.current_device() if use_gpu else None

        if not hasattr(self, "index_dataset"):
            self.index_dataset = Dataset.from_dict({"pids": range(self.ntotal)})
            self.index_dataset.load_faiss_index("embeddings", self.index_path / "index.faiss", device=device)

        self.encoder.cuda() if use_gpu else self.encoder.cpu()

        logger.debug(f"{rank}[{device}]> Encoding ...")
        Q = self.encoder.encode_queries(queries, progress_bar=progress_bar, to_cpu=True)

        logger.debug(f"{rank}[{device}]> Searching ...")
        result = self.index_dataset.search_batch("embeddings", Q.numpy(), k)

        return BatchResult(
            match_pids=result.total_indices,
            match_scores=result.total_scores,
        )
