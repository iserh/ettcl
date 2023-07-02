from ettcl.indexing.indexer import Indexer, IndexPath
from ettcl.encoding.encoder import Encoder, MultiProcessedEncoder
from datasets import Dataset
from ettcl.utils.utils import Devices, to_gpu_list
from os import PathLike
from pathlib import Path
import torch
from ettcl.indexing.faiss.configuration import FaissIndexerConfig
import json


class FaissSingleVectorIndexer(Indexer):

    def __init__(self, encoder: Encoder, config: FaissIndexerConfig = FaissIndexerConfig()) -> None:
        self.encoder = encoder
        self.config = config

    def index(
        self,
        output_path: str | PathLike,
        collection: list[str],
        gpus: Devices = True,
        n_processes: int = -1,
    ) -> IndexPath:
        output_path = Path(output_path)
        gpus = to_gpu_list(gpus)

        # setup number of multiprocessing processes
        if len(gpus):
            # set n_proc to number of gpus at max
            n_processes = min(n_processes, len(gpus)) if n_processes > 0 else len(gpus)
            n_processes = n_processes if n_processes > 1 else None
            device = None if n_processes > 1 else gpus[0]

        else:
            # default value for n_proc is to run single processed (=1)
            n_processes = n_processes if n_processes > 0 else None
            device = None

        with torch.cuda.device(device):
            encoder = MultiProcessedEncoder(self.encoder) if n_processes > 1 else self.encoder

            collection: Dataset = Dataset.from_dict({"text": collection})
            collection = collection.map(
                encoder.encode_passages,
                input_columns="text",
                batched=True,
                batch_size=2048,
                with_rank=n_processes is not None,
                num_proc=n_processes,
                desc="Encoding"
            )

            collection.set_format("numpy")
            collection = collection.add_faiss_index(
                column="embeddings",
                index_name="embeddings",
                device=gpus,
                string_factory=self.config.string_factory,
                train_size=self.config.train_size or len(collection),
                metric_type=self.config.metric_type_faiss,
            )


        output_path.mkdir(exist_ok=True, parents=True)

        metadata = {"ntotal": collection.get_index("embeddings").faiss_index.ntotal}
        with open(output_path / "metadata.json", "w") as fp:
            json.dump(metadata, fp)

        collection.save_faiss_index("embeddings", output_path / "index.faiss")

        return IndexPath(str(output_path))
