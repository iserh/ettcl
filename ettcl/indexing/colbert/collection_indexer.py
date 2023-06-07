from colbert.data.collection import Collection
from colbert.indexing.collection_indexer import CollectionIndexer as __CollectionIndexer
from colbert.indexing.index_saver import IndexSaver
from colbert.infra.config.config import ColBERTConfig
from colbert.infra.launcher import print_memory_stats

from ettcl.encoding.base_encoder import EncoderFactory


def index(
    config: ColBERTConfig,
    encoder_factory: EncoderFactory,
    collection: Collection | list[str] | str,
    shared_lists: list[list],
) -> None:
    indexer = CollectionIndexer(
        config=config, encoder_factory=encoder_factory, collection=collection
    )
    indexer.run(shared_lists)


class CollectionIndexer(__CollectionIndexer):
    def __init__(
        self,
        config: ColBERTConfig,
        encoder_factory: EncoderFactory,
        collection: Collection | list[str] | str,
    ) -> None:
        self.config = config
        self.rank, self.nranks = self.config.rank, self.config.nranks

        self.use_gpu = self.config.total_visible_gpus > 0

        if self.config.rank == 0:
            self.config.help()

        self.collection = Collection.cast(collection)
        self.encoder = encoder_factory.create(use_gpu=self.use_gpu)
        self.saver = IndexSaver(config)

        print_memory_stats(f"RANK:{self.rank}")
