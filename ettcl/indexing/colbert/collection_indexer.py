from colbert.data.collection import Collection
from colbert.indexing.collection_indexer import CollectionIndexer
from colbert.indexing.index_saver import IndexSaver

from ettcl.encoding.encoder import Encoder
from ettcl.indexing.colbert.settings import _IndexerSettings


def index(
    config: _IndexerSettings,
    encoder: Encoder,
    collection: list[str],
    shared_lists: list[list],
) -> None:
    indexer = CollectionIndexerWrapper(config=config, encoder=encoder, collection=collection)
    indexer.run(shared_lists)
    encoder.cpu()


class CollectionIndexerWrapper(CollectionIndexer):
    def __init__(
        self,
        config: _IndexerSettings,
        encoder: Encoder,
        collection: list[str],
    ) -> None:
        self.config = config
        self.rank, self.nranks = self.config.rank, self.config.nranks

        self.use_gpu = len(self.config.gpus_) > 0

        if self.config.rank == 0:
            self.config.help()

        self.collection = Collection.cast(collection)
        self.encoder = encoder
        if self.use_gpu:
            self.encoder = encoder.cuda()

        self.saver = IndexSaver(config)
