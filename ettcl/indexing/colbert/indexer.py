import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import torch
from colbert.infra.config.config import RunConfig
from colbert.infra.launcher import Launcher
from colbert.infra.run import Run

from ettcl.encoding.encoder import Encoder
from ettcl.indexing.colbert.collection_indexer import index
from ettcl.indexing.colbert.settings import _IndexerSettings
from ettcl.indexing.indexer import Indexer, IndexPath
from ettcl.utils.utils import Devices, to_gpu_list

logger = getLogger(__name__)


@dataclass
class ColBERTIndexerConfig:
    nbits: int = 2
    kmeans_niters: int = 4
    num_partitions: int | None = None


class ColBERTIndexer(Indexer):
    def __init__(self, encoder: Encoder, config: ColBERTIndexerConfig = ColBERTIndexerConfig()) -> None:
        self.encoder = encoder
        self.config = config

    def index(
        self,
        output_path: str | Path,
        collection: list[str],
        gpus: Devices = True,
        n_processes: int = -1,
        resume: bool = False,
    ) -> IndexPath:
        output_path = Path(output_path)
        gpus = to_gpu_list(gpus)
        use_gpu = len(gpus) > 0

        # setup number of multiprocessing processes
        if use_gpu:
            # set n_proc to number of gpus at max
            n_processes = min(n_processes, len(gpus)) if n_processes > 0 else len(gpus)
            if n_processes == 1:
                # store current device and set the selected one as default
                prev_device = torch.cuda.current_device()
                torch.cuda.set_device(gpus[0])
        else:
            # default value for n_proc is to run single processed (=1)
            n_processes = n_processes if n_processes > 0 else 1

        with Run().context(RunConfig(nranks=(max(1, n_processes)), gpus=gpus)):
            config = _IndexerSettings.from_existing(Run().config)
            config.configure(
                resume=resume,
                index_path=str(output_path),
                nbits=self.config.nbits,
                dim=self.encoder.embedding_dim,
                bsize=512,
                num_partitions=self.config.num_partitions,
            )

            output_path.mkdir(parents=True, exist_ok=True)
            if not resume:
                self.erase(output_path)

            if n_processes > 1:
                # launch multiprocessed
                self.__launch(collection, config)
            else:
                # launch as single process
                index(config, self.encoder, collection, [list()])
                if use_gpu:
                    # set back to previous device
                    torch.cuda.set_device(prev_device)

        return IndexPath(output_path)

    def __launch(self, collection: list[str], config: _IndexerSettings) -> None:
        manager = mp.Manager()
        shared_lists = [manager.list() for _ in range(config.nranks)]

        # Encodes collection into index using the CollectionIndexer class
        launcher = Launcher(index)
        launcher.launch(config, self.encoder, collection, shared_lists)

    @staticmethod
    def erase(index_path: str) -> list[str]:
        deleted = []
        for filename in sorted(os.listdir(index_path)):
            filename = os.path.join(index_path, filename)

            delete = filename.endswith(".json")
            delete = delete and ("metadata" in filename or "doclen" in filename or "plan" in filename)
            delete = delete or filename.endswith(".pt")

            if delete:
                deleted.append(filename)

        if len(deleted):
            logger.warning(f"#> Will delete {len(deleted)} files already at {index_path}")
            # time.sleep(10)

            for filename in deleted:
                os.remove(filename)

        return deleted
