#!/usr/bin/env python
import os
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainingArguments

from ettcl.core.triple_sampling import (
    DataCollatorForTriples,
    TripleSamplerDataset,
    TripleSamplingData,
)
from datasets import load_dataset
from ettcl.encoding import ColBERTEncoder
from ettcl.indexing import ColBERTIndexer, ColBERTIndexerConfig
from ettcl.modeling import ColBERTConfig, ColBERTForReranking, ColBERTModel, ColBERTTokenizer
from ettcl.searching import ColBERTSearcher, ColBERTSearcherConfig
from ettcl.utils.multiprocessing import run_multiprocessed
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb
from logging import getLogger

logger = getLogger(__name__)


def setup_configs_for_colbert(
    pretrained_model_name_or_path: str | os.PathLike,
    model_kwargs: dict[str],
    training_kwargs: dict[str],
    indexing_kwargs: dict[str] | None = None,
    searching_kwargs: dict[str] | None = None,
    searching_kwargs_sampling: dict[str] | None = None,
    **unused_kwargs,
) -> dict[str]:
    model_config = ColBERTConfig.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
    training_args = TrainingArguments(**training_kwargs)

    if indexing_kwargs is not None:
        indexer_config = ColBERTIndexerConfig(**indexing_kwargs)
    else:
        indexer_config = None

    if searching_kwargs is not None:
        searcher_config = ColBERTSearcherConfig(**searching_kwargs)
    else:
        searcher_config = None

    if searching_kwargs_sampling is not None:
        searcher_config_sampling = ColBERTSearcherConfig(**searching_kwargs_sampling)
    else:
        searcher_config_sampling = None

    return {
        "model_config": model_config,
        "training_args": training_args,
        "indexer_config": indexer_config,
        "searcher_config": searcher_config,
        "searcher_config_sampling": searcher_config_sampling,
    }


def evaluate(train_dataset, test_dataset, searcher, ks: list[int], label_column: str = "label", text_column: str = "text"):
    max_k = max(ks)
    print("Searching Eval")
    test_dataset.set_format(None)  # crucial to avoid leaked semaphore objects in map multiprocessing
    test_dataset = test_dataset.map(
        run_multiprocessed(searcher.search),
        input_columns=text_column,
        fn_kwargs={"k": max_k},
        batched=True,
        num_proc=torch.cuda.device_count(),
        with_rank=True,
        desc="Searching",
    )

    train_dataset.set_format("pt")
    test_dataset.set_format("pt")

    match_pids = test_dataset["match_pids"]
    if isinstance(match_pids, list):
        logger.warning(f"Fewer elements than k={max_k} matched, filling up with (-1).")
        match_pids = torch.nn.utils.rnn.pad_sequence(match_pids, batch_first=True, padding_value=-1)

    match_labels = train_dataset[label_column][match_pids.tolist()]

    print("Compute Metrics")
    metrics = {}
    for k in ks:
        knn = match_labels[:, :k]
        y_pred = torch.mode(knn)[0]
        assert -1 not in y_pred, "Not enough matches"

        metrics[f"accuracy/{k}"] = accuracy_score(y_pred=y_pred, y_true=test_dataset[label_column])
        metrics[f"precision/micro/{k}"] = precision_score(y_pred=y_pred, y_true=test_dataset[label_column], average="micro")
        metrics[f"precision/macro/{k}"] = precision_score(y_pred=y_pred, y_true=test_dataset[label_column], average="macro")
        metrics[f"recall/micro/{k}"] = recall_score(y_pred=y_pred, y_true=test_dataset[label_column], average="micro")
        metrics[f"recall/macro/{k}"] = recall_score(y_pred=y_pred, y_true=test_dataset[label_column], average="macro")
        metrics[f"f1/micro/{k}"] = f1_score(y_pred=y_pred, y_true=test_dataset[label_column], average="micro")
        metrics[f"f1/macro/{k}"] = f1_score(y_pred=y_pred, y_true=test_dataset[label_column], average="macro")

    print(metrics)
    wandb.run.log(metrics)


def train_colbert(
    dataset_name_or_path: str | os.PathLike,
    pretrained_model_name_or_path: str | os.PathLike,
    model_config: ColBERTConfig,
    training_args: TrainingArguments,
    tokenizer_kwargs: dict[str],
    sampling_kwargs: dict[str],
    do_eval: bool = True,
    do_test: bool = False,
    eval_ks: list[int] = [1],
    rebuild_index_interval: int = 1,
    indexer_config: ColBERTIndexerConfig | None = None,
    searcher_config: ColBERTSearcherConfig | None = None,
    searcher_config_sampling: ColBERTSearcherConfig | None = None,
    searcher_k_sampling: int | None = None,
    subsample_train: int | bool = False,
    subsample_test: int | bool = False,
    label_column: str = "label",
    text_column: str = "text",
    remove_columns: list[str] = [],
    freeze_base_model: bool = False,
    config: dict | None = None,
    **unused_kwargs,
) -> None:
    model = ColBERTForReranking.from_pretrained(pretrained_model_name_or_path, config=model_config)
    tokenizer = ColBERTTokenizer.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)
    encoder = ColBERTEncoder(model, tokenizer)
    indexer = ColBERTIndexer(encoder, indexer_config)

    output_dir = Path(training_args.output_dir)
    num_train_epochs = training_args.num_train_epochs
    do_searched_sampling = sampling_kwargs.get("sampling_method", "random") == "searched"
    assert (
        not do_searched_sampling or (searcher_config_sampling is not None and searcher_k_sampling is not None)
    ), "Searcher Config required when performing `searched` sampling."

    train_dataset = load_dataset(dataset_name_or_path, split="train").remove_columns(remove_columns)

    if do_test:
        raise NotImplementedError()
    if do_eval:
        train_dev_dataset = train_dataset.train_test_split(0.2, stratify_by_column=label_column)
        train_dataset = train_dev_dataset["train"]
        dev_dataset = train_dev_dataset["test"]
    if subsample_train is not False:
        train_dataset_ = train_dataset

    sampling_data_builder = TripleSamplingData(train_dataset[label_column], **sampling_kwargs)
    data_collator_for_triples = DataCollatorForTriples(tokenizer)
    if do_searched_sampling:
        sampling_input_columns = ["match_pids", "match_scores", label_column]
        report_missing_values_in_sampling = True
    else:
        report_missing_values_in_sampling = False
        sampling_input_columns = [label_column]

    # START WANDB
    wandb.init(dir=output_dir, config=config)

    # INITIAL SUBSAMPLING AND INDEX CREATION

    if subsample_train is not False:
            print("Subsample Train Dataset")
            train_dataset = train_dataset_.shuffle().select(range(subsample_train))

    if do_searched_sampling or do_eval:
        print("build initial index")
        index_path = output_dir / f"checkpoint-0" / "index"
        model.__class__ = ColBERTModel  # cast to ColBERTModel
        indexer.index(index_path, train_dataset[text_column], gpus=True)
    if do_searched_sampling:
        searcher_sampling = ColBERTSearcher(index_path, encoder, searcher_config_sampling)
    if do_eval:
        searcher = ColBERTSearcher(index_path, encoder, searcher_config)
        evaluate(train_dataset, dev_dataset, searcher, eval_ks, label_column, text_column)

    # TRAIN LOOP

    epoch, global_step = 0, 0
    while epoch < num_train_epochs:
        print(f"\n\n## EPOCH {epoch}\n")

        sampling_dataset = train_dataset
        sampling_dataset.set_format(None)
        if do_searched_sampling:
            print("Searching TRAIN")
            sampling_dataset = sampling_dataset.map(
                run_multiprocessed(searcher_sampling.search),
                input_columns=text_column,
                fn_kwargs={"k": searcher_k_sampling},
                batched=True,
                num_proc=torch.cuda.device_count(),
                with_rank=True,
                desc="Searching",
            )

        print("Create Sampling data")
        sampling_dataset = sampling_dataset.map(
            sampling_data_builder,
            input_columns=sampling_input_columns,
            fn_kwargs={"return_missing": report_missing_values_in_sampling},
            with_indices=True,
            remove_columns=sampling_input_columns,
        )

        if report_missing_values_in_sampling:
            missing_total = sum(sampling_dataset["missing"])
            sampling_dataset = sampling_dataset.remove_columns("missing")
            print(f"Missing {missing_total} values in sampling.")

        print("Tokenize")
        sampling_dataset = sampling_dataset.map(
            lambda batch: tokenizer(batch, truncation=True),
            input_columns=text_column,
            remove_columns=text_column,
            batched=True
        )

        triple_sampler_dataset = TripleSamplerDataset(sampling_dataset, sampling_data_builder.nway)

        print("Create Trainer")
        training_args.num_train_epochs = min(epoch + rebuild_index_interval, num_train_epochs)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=triple_sampler_dataset,
            data_collator=data_collator_for_triples,
        )

        print("Train")
        model.__class__ = ColBERTForReranking  # cast to ColBERTForReranking
        if freeze_base_model:
            for param in getattr(model, model.base_model_prefix).parameters():
                param.requires_grad = False
        trainer.train(resume_from_checkpoint=(epoch > 0))  # don't resume in the first epoch

        global_step = trainer.state.global_step
        epoch = trainer.state.epoch
        tokenizer.save_pretrained(output_dir / f"checkpoint-{global_step}")

        if subsample_train is not False:
            print("Subsample Train Dataset")
            train_dataset = train_dataset_.shuffle().select(range(subsample_train))

        if do_searched_sampling or do_eval:
            print("Rebuild index")
            index_path = output_dir / f"checkpoint-{global_step}" / "index"
            model.__class__ = ColBERTModel  # cast to ColBERTModel
            indexer.index(index_path, train_dataset[text_column], gpus=True)
            searcher.index_path = index_path
        if do_eval:
            evaluate(train_dataset, dev_dataset, searcher, eval_ks, label_column, text_column)

    # print("Save latest model")
    # trainer.save_model(output_dir / "latest")
    # tokenizer.tokenizer.save_pretrained(output_dir / f"latest")

    # print("Build latest index")
    # # for indexing/searching cast model to basic ColBERTModel
    # model.__class__ = ColBERTModel
    # indexer.index(output_dir / f"latest" / "index", train_dataset[text_column], gpus=True)


if __name__ == "__main__":
    import yaml

    with open("configs/training_config.yml", "r") as f:
        config = yaml.load(f, yaml.SafeLoader)

    config["training_kwargs"]["output_dir"] = os.path.join(
        "training",
        os.path.basename(config["dataset_name_or_path"]),
        os.path.basename(config["pretrained_model_name_or_path"]),
    )

    setup_configs = setup_configs_for_colbert(**config)
    train_colbert(**setup_configs, **config, config=config)
