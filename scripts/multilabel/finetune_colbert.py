#!/usr/bin/env python
import os
from dataclasses import asdict
from datetime import datetime

from datasets import load_from_disk
from transformers import TrainingArguments

from ettcl.core.reranking_mlc import RerankMLCTrainer, RerankMLCTrainerConfig
from ettcl.encoding import ColBERTEncoder
from ettcl.indexing import ColBERTIndexer, ColBERTIndexerConfig
from ettcl.logging import configure_logger
from ettcl.modeling import ColBERTConfig, ColBERTForReranking, ColBERTTokenizer
from ettcl.searching import ColBERTSearcher, ColBERTSearcherConfig
from ettcl.utils import seed_everything


def main(params: dict, log_level: str | int = "INFO") -> None:
    configure_logger(log_level)

    seed = params["seed"]["value"] if "seed" in params.keys() else 12345
    seed_everything(seed)

    output_dir = os.path.join(
        "training",
        os.path.basename(params["dataset"]["value"]),
        "colbert",
        os.path.basename(params["model"]["value"]),
        datetime.now().isoformat(),
    )

    model_config = ColBERTConfig.from_pretrained(params["model"]["value"], **params["model_config"]["value"])
    model = ColBERTForReranking.from_pretrained(params["model"]["value"], config=model_config)
    tokenizer = ColBERTTokenizer.from_pretrained(params["model"]["value"], **params["tokenizer"]["value"])
    encoder = ColBERTEncoder(model.colbert, tokenizer)

    indexer_config = ColBERTIndexerConfig(**params["indexer"]["value"])
    indexer = ColBERTIndexer(encoder, indexer_config)

    if "searcher_eval" in params.keys():
        searcher_eval_config = ColBERTSearcherConfig(**params["searcher_eval"]["value"])
        searcher_eval = ColBERTSearcher(None, encoder, searcher_eval_config)
    else:
        searcher_eval = None

    if "searcher_sampling" in params.keys():
        searcher_sampling_config = ColBERTSearcherConfig(**params["searcher_sampling"]["value"])
        searcher_sampling = ColBERTSearcher(None, encoder, searcher_eval_config)
    else:
        searcher_sampling = None

    params["training"]["value"].pop("output_dir", None)
    params["training"]["value"].pop("seed", None)
    training_args = TrainingArguments(output_dir=output_dir, seed=seed, **params["training"]["value"])
    config = RerankMLCTrainerConfig(**params["config"]["value"])

    dataset = load_from_disk(params["dataset"]["value"])
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    trainer = RerankMLCTrainer(
        model=model,
        encoder=encoder,
        tokenizer=tokenizer,
        config=config,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if config.do_eval else None,
        indexer=indexer,
        searcher_eval=searcher_eval,
        searcher_sampling=searcher_sampling,
    )

    trainer.run_config = {
        "seed": seed,
        "dataset": params["dataset"]["value"],
        "architecture": "ColBERT",
        "model": params["model"]["value"],
        "model_config": model_config.to_dict(),
        "tokenizer": tokenizer.init_kwargs,
        "indexer": asdict(indexer_config),
        "training": training_args.to_dict(),
        "config": asdict(config),
    }

    if searcher_eval is not None:
        trainer.run_config.update({"searcher_eval": asdict(searcher_eval_config)})

    if searcher_sampling is not None:
        trainer.run_config.update({"searcher_sampling": asdict(searcher_sampling_config)})

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
