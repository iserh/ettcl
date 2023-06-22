from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from colbert.infra.config import BaseConfig, RunConfig, RunSettings, SearchSettings
from colbert.infra.run import Run
from colbert.search.index_storage import IndexScorer
from colbert.utils.utils import zipstar
from ettcl.logging.tqdm import trange

from ettcl.encoding.encoder import Encoder
from ettcl.searching.searcher import Searcher, SearchResult, TensorType, TextQueries


@dataclass
class ColBERTSearcherConfig:
    ncells: int | None = None


@dataclass
class _SearcherSettings(RunSettings, SearchSettings, BaseConfig):
    interaction: str = "colbert"


class ColBERTSearcher(Searcher):
    def __init__(
        self,
        index_path: str,
        encoder: Encoder,
        config: ColBERTSearcherConfig = ColBERTSearcherConfig(),
    ) -> None:
        self.encoder = encoder
        self.config = config
        self.index_path = index_path

    def search(
        self,
        queries: TextQueries,
        k: int = 10,
        return_tensors: bool | str | TensorType = "pt",
        use_gpu: bool = True,
        progress_bar: bool = True,
        rank: int | str = "#",
    ) -> SearchResult:
        match queries:
            case str():
                queries = [queries]
            case dict():
                queries = queries.values()

        # disable progress bar in multiprocess
        progress_bar = progress_bar if rank == "#" else False

        use_gpu = use_gpu and torch.cuda.is_available()
        run_config = RunConfig(gpus=None if use_gpu else 0)

        if not hasattr(self, "ranker") or self.ranker.index_path != self.index_path:
            self.ranker = IndexScorer(self.index_path, use_gpu)

        if use_gpu:
            self.encoder.cuda()
            self.ranker.cuda()
        else:
            self.encoder.cpu()
            self.ranker.cpu()

        with Run().context(run_config):

            _config = _SearcherSettings.from_existing(Run().config)
            _config.configure(ncells=self.config.ncells)

            print(f"{rank}[{self.ranker.device}]> Searching ...")
            Q = self.encoder.encode_queries(queries, progress_bar=progress_bar)
            return self._search_all_Q(Q, k, _config, return_tensors, progress_bar)

    def _search_all_Q(
        self,
        Q: torch.Tensor,
        k: int,
        args: _SearcherSettings,
        return_tensors: bool | str | TensorType,
        progress_bar: bool = True,
    ) -> SearchResult:
        trange_ = trange if progress_bar else range
        pids, scores = zipstar(
            [
                self.dense_search(Q[query_idx : query_idx + 1], k, args, return_tensors)
                for query_idx in trange_(Q.size(0))
            ]
        )

        return {"match_pids": pids, "match_scores": scores}

    def dense_search(
        self,
        Q: torch.Tensor,
        k: int,
        args: _SearcherSettings,
        return_tensors: bool | str | TensorType = "pt",
    ) -> dict[str, list | torch.Tensor | np.ndarray]:
        if k <= 10:
            if args.ncells is None:
                args.configure(ncells=1)
            if args.centroid_score_threshold is None:
                args.configure(centroid_score_threshold=0.5)
            if args.ndocs is None:
                args.configure(ndocs=256)
        elif k <= 100:
            if args.ncells is None:
                args.configure(ncells=2)
            if args.centroid_score_threshold is None:
                args.configure(centroid_score_threshold=0.45)
            if args.ndocs is None:
                args.configure(ndocs=1024)
        else:
            if args.ncells is None:
                args.configure(ncells=4)
            if args.centroid_score_threshold is None:
                args.configure(centroid_score_threshold=0.4)
            if args.ndocs is None:
                args.configure(ndocs=max(k * 4, 4096))

        pids, scores = self.ranker.rank(args, Q)

        # move to cpu
        pids, scores = pids[:k].cpu(), scores[:k].cpu()

        match return_tensors:
            case False:
                return pids.tolist(), scores.tolist()
            case True | "pt":
                return pids, scores
            case "np":
                return pids.numpy(), scores.numpy()
            case _:
                raise NotImplementedError()
