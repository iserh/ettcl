from dataclasses import dataclass

import torch
from colbert.infra.config import BaseConfig, RunConfig, RunSettings, SearchSettings
from colbert.infra.run import Run
from colbert.search.index_storage import IndexScorer
from colbert.utils.utils import zipstar
from tqdm import trange

from ettcl.encoding.base_encoder import BaseEncoder
from ettcl.searching.base_searcher import BaseSearcher, SearcherConfig, SearchResult, TextQueries, TensorType


@dataclass
class _SearcherSettings(RunSettings, SearchSettings, BaseConfig):
    interaction: str = "colbert"


class ColBERTSearcher(BaseSearcher):
    def __init__(
        self, index_path: str, encoder: BaseEncoder, args: SearcherConfig = SearcherConfig()
    ) -> None:
        super().__init__(index_path, encoder, args)

    def search(
        self,
        queries: TextQueries,
        k: int,
        return_tensors: bool | str | TensorType = "pt",
        gpu: bool | int = True,
        progress_bar: bool = False,
    ) -> SearchResult:
        match queries:
            case str():
                queries = [queries]
            case dict():
                queries = list(queries.values())

        use_gpu = torch.cuda.is_available() and gpu is not False
        self.gpu = gpu if type(gpu) == int else 0

        if use_gpu:
            torch.cuda.set_device(self.gpu)
            print(f"USING DEVICE {torch.cuda.current_device()}")

        self.ranker = IndexScorer(self.index_path, use_gpu=use_gpu)
        self.encoder.cuda() if use_gpu else self.encoder.cpu()

        with Run().context(RunConfig(gpus=0 if not use_gpu else None)):
            config = _SearcherSettings.from_existing(Run().config)
            config.configure(ncells=self.args.ncells)

            Q = self.encoder.encode_queries(queries)
            return self._search_all_Q(Q, k, config, return_tensors, progress_bar)

    def _search_all_Q(
        self,
        Q: torch.Tensor,
        k: int,
        config: _SearcherSettings,
        return_tensors: bool | str | TensorType,
        progress_bar: bool = False,
    ) -> SearchResult:
        results = []
        for query_idx in trange(Q.size(0), disable=not progress_bar, desc=f"Searching (device {self.gpu})"):
            result = self.dense_search(Q[query_idx : query_idx + 1], k, config, return_tensors)
            results.append(result)

        return zipstar(results)
        # return zipstar(
        #     [
        #         self.dense_search(Q[query_idx : query_idx + 1], k, config, return_tensors)
        #         for query_idx in trange(Q.size(0), disable=not progress_bar, desc=f"Searching (device {self.gpu})")
        #     ]
        # )

    def dense_search(
        self,
        Q: torch.Tensor,
        k: int,
        config: _SearcherSettings,
        return_tensors: bool | str | TensorType = "pt",
    ) -> SearchResult:
        if k <= 10:
            if config.ncells is None:
                config.configure(ncells=1)
            if config.centroid_score_threshold is None:
                config.configure(centroid_score_threshold=0.5)
            if config.ndocs is None:
                config.configure(ndocs=256)
        elif k <= 100:
            if config.ncells is None:
                config.configure(ncells=2)
            if config.centroid_score_threshold is None:
                config.configure(centroid_score_threshold=0.45)
            if config.ndocs is None:
                config.configure(ndocs=1024)
        else:
            if config.ncells is None:
                config.configure(ncells=4)
            if config.centroid_score_threshold is None:
                config.configure(centroid_score_threshold=0.4)
            if config.ndocs is None:
                config.configure(ndocs=max(k * 4, 4096))

        pids, scores = self.ranker.rank(config, Q)

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
