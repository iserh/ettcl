#!/usr/bin/env python
import os
import shutil
from dataclasses import asdict
from datetime import datetime

from datasets import load_dataset

from ettcl.core.evaluate import Evaluator, EvaluatorConfig
from ettcl.encoding import ColBERTEncoder
from ettcl.indexing import ColBERTIndexer, ColBERTIndexerConfig
from ettcl.logging import configure_logger
from ettcl.modeling import ColBERTConfig, ColBERTModel, ColBERTTokenizer
from ettcl.searching import ColBERTSearcher, ColBERTSearcherConfig
from ettcl.utils import seed_everything


def main() -> None:
    dataset_name = "imdb"
    model_name = "bert-base-uncased"

    index_path = os.path.join(
        "indices",
        os.path.basename(dataset_name),
        os.path.basename(model_name),
    )

    model_config = ColBERTConfig.from_pretrained(model_name)
    encoder = ColBERTEncoder.from_pretrained(model_name, model_kwargs={"config": model_config})

    indexer_config = ColBERTIndexerConfig(
        nbits=2,
        num_partitions_fac=20,
    )
    indexer = ColBERTIndexer(encoder, indexer_config)

    searcher_config = ColBERTSearcherConfig(**params["searcher"]["value"])
    searcher = ColBERTSearcher(None, encoder, searcher_config)

    config = EvaluatorConfig(output_dir, **params["config"]["value"])

    train_dataset = load_dataset(params["dataset"]["value"], split="train")
    test_dataset = load_dataset(params["dataset"]["value"], split="test")

    evaluator = Evaluator(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        config=config,
        encoder=encoder,
        indexer=indexer,
        searcher=searcher,
    )

    evaluator.run_config = {
        "dataset": params["dataset"]["value"],
        "model": params["model"]["value"],
        "seed": seed,
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.init_kwargs,
        "indexer": asdict(indexer_config),
        "config": asdict(config),
    }

    evaluator.evaluate()


if __name__ == "__main__":
    main()
