#!/usr/bin/env python
import os
import shutil
from dataclasses import asdict
from datetime import datetime

from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from ettcl.core.reranking import RerankTrainer, RerankTrainerConfig
from ettcl.encoding import STEncoder
from ettcl.indexing import FaissIndexerConfig, FaissSingleVectorIndexer
from ettcl.logging import configure_logger
from ettcl.modeling import SentenceTransformerForReranking
from ettcl.searching import FaissSingleVectorSearcher
from ettcl.utils import seed_everything


def main(params: dict, log_level: str | int = "INFO") -> None:
    configure_logger(log_level)

    seed = params["seed"]["value"] if "seed" in params.keys() else 12345
    seed_everything(seed)

    output_dir = os.path.join(
        "training",
        os.path.basename(params["dataset"]["value"]),
        os.path.basename(params["model"]["value"]),
        datetime.now().isoformat(),
    )

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    model = SentenceTransformerForReranking(params["model"]["value"])
    # workaround: sentence-transformer tokenizer has wrong model_max_length
    model.sentence_transformer[0].tokenizer = AutoTokenizer.from_pretrained(
        params["model"]["value"], **params.get("tokenizer", {}).get("value", {})
    )
    encoder = STEncoder(model.sentence_transformer, normalize_embeddings=True)

    indexer_config = FaissIndexerConfig(**params["indexer"]["value"])
    indexer = FaissSingleVectorIndexer(encoder, indexer_config)

    searcher = FaissSingleVectorSearcher(None, encoder)

    params["training"]["value"].pop("output_dir", None)
    training_args = TrainingArguments(output_dir=output_dir, seed=seed, **params["training"]["value"])
    config = RerankTrainerConfig(**params["config"]["value"])

    train_dataset = load_dataset(params["dataset"]["value"], split="train")
    test_dataset = load_dataset(params["dataset"]["value"], split="test") if config.do_eval else None

    trainer = RerankTrainer(
        trainer_cls=Trainer,
        model=model,
        encoder=encoder,
        tokenizer=model.sentence_transformer.tokenizer,
        config=config,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        indexer=indexer,
        searcher_eval=searcher,
        searcher_sampling=searcher,
    )

    trainer.run_config = {
        "dataset": params["dataset"]["value"],
        "architecture": "S-BERT",
        "model": params["model"]["value"],
        "model_config": model.config.to_dict(),
        "tokenizer": model.sentence_transformer.tokenizer.init_kwargs,
        "indexer": asdict(indexer_config),
        "training": training_args.to_dict(),
        "config": asdict(config),
    }

    trainer.train()


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
