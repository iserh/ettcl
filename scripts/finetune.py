#!/usr/bin/env python
import os
from dataclasses import asdict

from datasets import load_dataset
from transformers import TrainingArguments

from ettcl.core.reranking import RerankTrainer, RerankTrainerConfig
from ettcl.encoding import ColBERTEncoder
from ettcl.indexing import ColBERTIndexer, ColBERTIndexerConfig
from ettcl.modeling import ColBERTConfig, ColBERTModel, ColBERTTrainer, ColBERTTokenizer
from ettcl.searching import ColBERTSearcher, ColBERTSearcherConfig


def main(params: dict) -> None:
    output_dir = os.path.join(
        "training",
        os.path.basename(params["dataset"]),
        os.path.basename(params["model"]),
    )

    model_config = ColBERTConfig.from_pretrained(params["model"], **params["model_config"])
    model = ColBERTModel.from_pretrained(params["model"], config=model_config)
    tokenizer = ColBERTTokenizer.from_pretrained(params["model"], **params["tokenizer"])
    encoder = ColBERTEncoder(model, tokenizer)

    indexer_config = ColBERTIndexerConfig(**params["indexer"])
    indexer = ColBERTIndexer(encoder, indexer_config)

    if params.get("searcher_eval", None) is not None:
        searcher_eval_config = ColBERTSearcherConfig(**params["searcher_eval"])
        searcher_eval = ColBERTSearcher(None, encoder, searcher_eval_config)
    else:
        searcher_eval = None

    if params.get("searcher_sampling", None) is not None:
        searcher_sampling_config = ColBERTSearcherConfig(**params["searcher_sampling"])
        searcher_sampling = ColBERTSearcher(None, encoder, searcher_eval_config)
    else:
        searcher_sampling = None

    training_args = TrainingArguments(output_dir=output_dir, **params["training"])
    config = RerankTrainerConfig(**params["config"])

    train_dataset = load_dataset(params["dataset"], split="train")
    test_dataset = load_dataset(params["dataset"], split="test") if config.do_eval else None

    trainer = RerankTrainer(
        trainer_cls=ColBERTTrainer,
        model=model,
        encoder=encoder,
        tokenizer=tokenizer,
        config=config,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        indexer=indexer,
        searcher_eval=searcher_eval,
        searcher_sampling=searcher_sampling,
    )

    trainer.run_config = {
        "dataset": params["dataset"],
        "model": params["model"],
        "model_config": model_config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "indexer": asdict(indexer_config),
        "training": asdict(training_args),
        "config": asdict(config),
    }

    if searcher_eval is not None:
        trainer.run_config.update({"searcher_eval": asdict(searcher_eval_config)})

    if searcher_sampling is not None:
        trainer.run_config.update({"searcher_sampling": asdict(searcher_sampling_config)})

    trainer.train()


if __name__ == "__main__":
    import yaml
    from ettcl.logging import configure_logger

    configure_logger("INFO")

    with open("configs/training_config.yml", "r") as f:
        params = yaml.load(f, yaml.SafeLoader)

    main(params)
