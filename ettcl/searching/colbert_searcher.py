from dataclasses import dataclass

import torch
from colbert.infra.config import BaseConfig, QuerySettings, RunConfig, RunSettings, SearchSettings
from colbert.infra.launcher import print_memory_stats
from colbert.infra.run import Run
from colbert.search.index_storage import IndexScorer
from colbert.utils.utils import zipstar
from tqdm import trange

from ettcl.encoding.base_encoder import EncoderFactory
from ettcl.searching.base_searcher import (
    BaseSearcher,
    SearchingArguments,
    SearchResult,
    TextQueries,
)


@dataclass
class SearcherConfig(RunSettings, SearchSettings, QuerySettings, BaseConfig):
    pass


class ColBERTSearcher(BaseSearcher):
    def __init__(
        self,
        index_path: str,
        encoder_factory: EncoderFactory,
        args: SearchingArguments = SearchingArguments(),
    ) -> None:
        super().__init__(index_path, encoder_factory, args)

        use_gpu = len(self.args.gpus_) > 0
        self.encoder = encoder_factory.create(use_gpu=use_gpu)
        self.ranker = IndexScorer(index_path, use_gpu)

        print_memory_stats()

    def search(self, queries: TextQueries, k: int) -> SearchResult:
        match queries:
            case str():
                queries = [queries]
            case dict():
                queries = list(queries.values())

        run_config = RunConfig(
            gpus=self.args.gpus_,
        )

        # because searcher is single threaded we need to set available cuda devices,
        # usually launcher takes care of this
        run_config.configure(total_visible_gpus=len(self.args.gpus_))

        with Run().context(run_config):
            config = SearcherConfig.from_existing(Run().config)

            Q = self.encoder.encode_queries(queries)
            return self._search_all_Q(Q, k, config)

    def _search_all_Q(
        self, Q: torch.Tensor, k: int, config: SearcherConfig
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        return zipstar(
            [
                self.dense_search(Q[query_idx : query_idx + 1], k, config)
                for query_idx in trange(Q.size(0))
            ]
        )

    def dense_search(self, Q: torch.Tensor, k: int, config: SearcherConfig) -> SearchResult:
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

        return pids[:k], scores[:k]
