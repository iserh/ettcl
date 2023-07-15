#!/usr/bin/env python
import os
from dataclasses import asdict
from datetime import datetime

from datasets import load_from_disk

from ettcl.core.evaluate_mlc import EvaluatorMLC, EvaluatorMLCConfig
from ettcl.encoding import ColBERTEncoder
from ettcl.indexing import ColBERTIndexer, ColBERTIndexerConfig
from ettcl.logging import configure_logger
from ettcl.modeling import SentenceColBERTModel, SentenceTokenizer
from ettcl.searching import ColBERTSearcher, ColBERTSearcherConfig
from ettcl.utils import seed_everything, split_into_sentences


def main(params: dict, log_level: str | int = "INFO") -> None:
    configure_logger(log_level)

    seed = params["seed"]["value"] if "seed" in params.keys() else 12345
    seed_everything(seed)

    if params["config"]["value"].get("output_dir", None) is not None:
        output_dir = params["config"]["value"].pop("output_dir")
    else:
        output_dir = os.path.join(
            "evaluation",
            os.path.basename(params["dataset"]["value"]),
            "scolbert",
            os.path.basename(params["model"]["value"]),
            datetime.now().isoformat(),
        )

    model = SentenceColBERTModel.from_pretrained(params["model"]["value"])
    tokenizer = SentenceTokenizer.from_pretrained(params["model"]["value"], **params["tokenizer"]["value"])
    encoder = ColBERTEncoder(model, tokenizer)

    indexer_config = ColBERTIndexerConfig(**params["indexer"]["value"])
    indexer = ColBERTIndexer(encoder, indexer_config)

    searcher_config = ColBERTSearcherConfig(**params["searcher_eval"]["value"])
    searcher = ColBERTSearcher(None, encoder, searcher_config)

    config = EvaluatorMLCConfig(output_dir, **params["config"]["value"])

    dataset = load_from_disk(params["dataset"]["value"])
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    num_sentences = params["num_sentences"]["value"]
    train_dataset = train_dataset.map(
        lambda text: {"sents": split_into_sentences(text, num_sentences)},
        input_columns="text",
        remove_columns="text",
        desc="split_into_sentences",
    )

    test_dataset = test_dataset.map(
        lambda text: {"sents": split_into_sentences(text, num_sentences)},
        input_columns="text",
        remove_columns="text",
        desc="split_into_sentences",
    )

    config.text_column = "sents"

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
        "num_sentences": num_sentences,
        "architecture": "S-ColBERT",
        "model": params["model"]["value"],
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.init_kwargs,
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
