import multiprocessing as mp
import os
import time
from pathlib import Path

from colbert.data.collection import Collection
from colbert.infra.config.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import Launcher
from colbert.infra.run import Run
from colbert.utils.utils import print_message

from ettcl.indexing.base_indexer import BaseIndexer
from ettcl.indexing.colbert.collection_indexer import index


class ColBERTIndexer(BaseIndexer):
    def index(
        self, index_path: str, collection: Collection | list[str] | str, resume: bool = False
    ) -> None:
        index_path = Path(index_path)

        run_config = RunConfig(
            nranks=self.args.nranks,
            gpus=self.args.gpus_,
        )

        # only way to disable colbert's gpu usage is to limit the visible gpus
        # (Usually the launcher does this by setting CUDA_VISIBLE_DEVICES)
        if len(self.args.gpus_) == 0:
            run_config.configure(total_visible_gpus=0)

        with Run().context(run_config):
            # if isinstance(self.encoder.config, ColBERTConfig):
            #     checkpoint_config = ColBERTConfig.from_existing(self.encoder.config)

            config = ColBERTConfig.from_existing(Run().config)

            config.configure(
                resume=resume,
                index_path=str(index_path),
                nbits=self.args.nbits,
                dim=self.args.dim,
                bsize=64,
                partitions=None,
            )

            index_path.mkdir(parents=True, exist_ok=True)
            if not resume:
                self.erase(index_path)

            if config.nranks > 1:
                self.__launch(collection, config)
            else:
                index(config, self.encoder_factory, collection, [list()])

    def __launch(self, collection: Collection | list[str], config: ColBERTConfig) -> None:
        manager = mp.Manager()
        shared_lists = [manager.list() for _ in range(config.nranks)]

        # Encodes collection into index using the CollectionIndexer class
        launcher = Launcher(index)
        launcher.launch(config, self.encoder_factory, collection, shared_lists)

    @staticmethod
    def erase(index_path: str) -> list[str]:
        deleted = []
        for filename in sorted(os.listdir(index_path)):
            filename = os.path.join(index_path, filename)

            delete = filename.endswith(".json")
            delete = delete and (
                "metadata" in filename or "doclen" in filename or "plan" in filename
            )
            delete = delete or filename.endswith(".pt")

            if delete:
                deleted.append(filename)

        if len(deleted):
            print_message(
                f"#> Will delete {len(deleted)} files already at {index_path} in 20 seconds..."
            )
            time.sleep(20)

            for filename in deleted:
                os.remove(filename)

        return deleted
