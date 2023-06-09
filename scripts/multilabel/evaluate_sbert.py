#!/usr/bin/env python
import os
from dataclasses import asdict
from datetime import datetime

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

from ettcl.core.evaluate_mlc import EvaluatorMLC, EvaluatorMLCConfig
from ettcl.encoding import STEncoder
from ettcl.indexing import FaissIndexerConfig, FaissSingleVectorIndexer
from ettcl.logging import configure_logger
from ettcl.searching import FaissSingleVectorSearcher
from ettcl.utils import seed_everything


def main(params: dict, log_level: str | int = "INFO") -> None:
    configure_logger(log_level)

    seed = params["seed"]["value"] if "seed" in params.keys() else 12345
    seed_everything(seed)

    output_dir = os.path.join(
        "evaluation",
        os.path.basename(params["dataset"]["value"]),
        "sbert",
        os.path.basename(params["model"]["value"]),
        datetime.now().isoformat(),
    )

    model = SentenceTransformer(params["model"]["value"])
    encoder = STEncoder(model, normalize_embeddings=True)

    indexer_config = FaissIndexerConfig(**params["indexer"]["value"])
    indexer = FaissSingleVectorIndexer(encoder, indexer_config)

    searcher = FaissSingleVectorSearcher(None, encoder)

    config = EvaluatorMLCConfig(output_dir, **params["config"]["value"])

    dataset = load_from_disk(params["dataset"]["value"])
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    evaluator = EvaluatorMLC(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        config=config,
        encoder=encoder,
        indexer=indexer,
        searcher=searcher,
    )

    evaluator.run_config = {
        "seed": seed,
        "dataset": params["dataset"]["value"],
        "architecture": "ColBERT",
        "model": params["model"]["value"],
        "model_config": model[0].auto_model.config.to_dict(),
        "tokenizer": model.tokenizer.init_kwargs,
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
