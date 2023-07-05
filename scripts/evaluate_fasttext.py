#!/usr/bin/env python
import os
import shutil
from dataclasses import asdict
from datetime import datetime

from datasets import load_dataset

from ettcl.core.evaluate import Evaluator, EvaluatorConfig
from ettcl.encoding import Word2VecEncoder
from ettcl.indexing import ColBERTIndexer, ColBERTIndexerConfig
from ettcl.logging import configure_logger
from gensim.models import FastText
from ettcl.searching import ColBERTSearcher, ColBERTSearcherConfig
from ettcl.utils import seed_everything


def main(params: dict, log_level: str | int = "INFO") -> None:
    configure_logger(log_level)

    seed = params["seed"]["value"] if "seed" in params.keys() else 12345
    seed_everything(seed)

    output_dir = os.path.join(
        "evaluation",
        os.path.basename(params["dataset"]["value"]),
        os.path.basename(params["model"]["value"]),
        datetime.now().isoformat(),
    )

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    model = FastText.load(params["model"]["value"])
    encoder = Word2VecEncoder(model)

    indexer_config = ColBERTIndexerConfig(**params["indexer"]["value"])
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
        "architecture": "FastText",
        "model": params["model"]["value"],
        "seed": seed,
        "indexer": asdict(indexer_config),
        "config": asdict(config),
    }

    evaluator.evaluate()


if __name__ == "__main__":
    from argparse import ArgumentParser

    import yaml

    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the train config yaml file.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    with open(args.path, "r") as f:
        params = yaml.load(f, yaml.SafeLoader)

    main(params, args.log_level)
