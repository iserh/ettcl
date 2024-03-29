from dataclasses import dataclass
from logging import getLogger

import torch
from colbert.infra.config import BaseConfig, RunConfig, RunSettings, SearchSettings
from colbert.infra.run import Run
from colbert.search.index_storage import IndexScorer
from colbert.utils.utils import zipstar

from ettcl.encoding.encoder import Encoder
from ettcl.indexing.indexer import IndexPath
from ettcl.logging.tqdm import trange
from ettcl.searching.searcher import BatchResult, Searcher, TensorLike, TensorType, TextQueries

logger = getLogger(__name__)


@dataclass
class ColBERTSearcherConfig:
    ncells: int | None = None
    centroid_score_threshold: float | None = None
    ndocs: int | None = None
    plaid_num_elem_batch: int = 3e8
    skip_plaid_stage_3: bool = False
    plaid_stage_2_3_cpu: bool = False
    filter_pids: list[int] | None = None


@dataclass
class _SearcherSettings(RunSettings, SearchSettings, BaseConfig):
    interaction: str = "colbert"
    plaid_num_elem_batch: int = 3e8
    skip_plaid_stage_3: bool = False
    plaid_stage_2_3_cpu: bool = False


class ColBERTSearcher(Searcher):
    def __init__(
        self,
        index_path: str | IndexPath,
        encoder: Encoder,
        config: ColBERTSearcherConfig = ColBERTSearcherConfig(),
    ) -> None:
        self.encoder = encoder
        self.config = config
        # We are using this IndexPath str wrapper to ensure that
        # huggingface datasets map method notices any file changes
        # in the index (e.g. when rebuilding the index)
        if index_path is not None:
            self.__index_path = IndexPath(index_path)
        else:
            self.__index_path = None

    @property
    def index_path(self) -> IndexPath:
        return self.__index_path

    @index_path.setter
    def index_path(self, path: str | IndexPath) -> None:
        self.__index_path = IndexPath(path)

    def search(
        self,
        queries: TextQueries,
        k: int = 10,
        return_tensors: bool | str | TensorType = "pt",
        use_gpu: bool = True,
        progress_bar: bool = True,
        rank: int | str = "#",
    ) -> BatchResult:
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

        if self.config.filter_pids is not None:
            filter_pids = torch.tensor(
                self.config.filter_pids, device=torch.cuda.current_device() if use_gpu else "cpu"
            )
            self.filter_fn = lambda pids: pids[~torch.isin(pids, filter_pids)]
        else:
            self.filter_fn = None

        with Run().context(run_config):
            _config = _SearcherSettings.from_existing(Run().config)
            _config.configure(
                ncells=self.config.ncells,
                centroid_score_threshold=self.config.centroid_score_threshold,
                ndocs=self.config.ndocs,
                plaid_num_elem_batch=self.config.plaid_num_elem_batch,
                skip_plaid_stage_3=self.config.skip_plaid_stage_3,
                plaid_stage_2_3_cpu=self.config.plaid_stage_2_3_cpu,
            )

            logger.debug(f"{rank}[{self.ranker.device}]> Encoding ...")
            Q = self.encoder.encode_queries(queries, progress_bar=progress_bar)
            logger.debug(f"{rank}[{self.ranker.device}]> Searching ...")
            return self._search_all_Q(Q, k, _config, return_tensors, progress_bar)

    def _search_all_Q(
        self,
        Q: list[torch.Tensor],
        k: int,
        args: _SearcherSettings,
        return_tensors: bool | str | TensorType,
        progress_bar: bool = True,
    ) -> BatchResult:
        pids, scores = zipstar(
            [
                self.dense_search(Q[query_idx], k, args, return_tensors)
                for query_idx in trange(len(Q), disable=not progress_bar)
            ]
        )
        return BatchResult(pids, scores)

    def dense_search(
        self,
        Q: torch.Tensor,
        k: int,
        args: _SearcherSettings,
        return_tensors: bool | str | TensorType = "pt",
    ) -> tuple[TensorLike, TensorLike]:
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
                args.configure(ndocs=max(k * 4, 1024))

        pids, scores = self.ranker.rank(args, Q.unsqueeze(0), self.filter_fn)

        # move to cpu
        if len(pids):
            pids, scores = pids[:k].cpu(), scores[:k].cpu().to(torch.float32)
        else:
            pids = torch.LongTensor([])
            scores = torch.FloatTensor([])

        match return_tensors:
            case True | "pt":
                return pids, scores
            case False:
                return pids.tolist(), scores.tolist()
            case "np":
                return pids.numpy(), scores.numpy()
            case _:
                raise NotImplementedError()
