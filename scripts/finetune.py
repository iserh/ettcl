#!/usr/bin/env python
import os
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from ettcl.core.triple_sampling import (
    DataCollatorForTriples,
    TripleSamplerDataset,
    TripleSamplingData,
)
from ettcl.encoding import ColBERTEncoder
from ettcl.indexing import ColBERTIndexer, ColBERTIndexerConfig
from ettcl.modeling import ColBERTConfig, ColBERTForReranking, ColBERTModel, ColBERTTokenizer
from ettcl.searching import ColBERTSearcher, ColBERTSearcherConfig
from ettcl.utils.multiprocessing import run_multiprocessed


def setup_configs_for_colbert(
    pretrained_model_name_or_path: str | os.PathLike,
    model_kwargs: dict[str],
    training_kwargs: dict[str],
    indexing_kwargs: dict[str],
    searching_kwargs: dict[str],
) -> dict[str]:
    model_config = ColBERTConfig.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
    training_args = TrainingArguments(**training_kwargs)
    indexer_config = ColBERTIndexerConfig(**indexing_kwargs)

    if searching_kwargs is not None:
        searcher_config = ColBERTSearcherConfig(**searching_kwargs)
    else:
        searcher_config = None

    return {
        "model_config": model_config,
        "training_args": training_args,
        "indexer_config": indexer_config,
        "searcher_config": searcher_config,
    }


def train_colbert(
    dataset_name_or_path: str | os.PathLike,
    pretrained_model_name_or_path: str | os.PathLike,
    model_config: ColBERTConfig,
    training_args: TrainingArguments,
    indexer_config: ColBERTIndexerConfig,
    tokenizer_kwargs: dict[str],
    sampling_kwargs: dict[str],
    rebuild_index_interval: int = 1,
    do_eval: bool = False,
    searcher_config: ColBERTSearcherConfig | None = None,
    searcher_k: int | None = None,
    subsample_train: int | bool = False,
    subsample_test: int | bool = False,
) -> None:
    model = ColBERTForReranking.from_pretrained(pretrained_model_name_or_path, config=model_config)
    tokenizer = ColBERTTokenizer.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)
    encoder = ColBERTEncoder(model, tokenizer)
    indexer = ColBERTIndexer(encoder, indexer_config)

    output_dir = Path(training_args.output_dir)
    num_train_epochs = training_args.num_train_epochs
    do_searched_sampling = sampling_kwargs.get("sampling_method", "random") == "searched"
    assert (
        do_searched_sampling == searcher_config is not None == searcher_k is not None
    ), "Searcher Config required when performing `searched` sampling."

    train_dataset = load_dataset(dataset_name_or_path, split="train")
    if do_eval:
        test_dataset = load_dataset(dataset_name_or_path, split="test")
    else:
        raise NotImplementedError()

    if subsample_train is not False:
        train_dataset_ = train_dataset
        train_dataset = train_dataset_.shuffle().take(subsample_train)

    print("Build initial index")
    model.__class__ = ColBERTModel  # cast to ColBERTModel
    index_path = output_dir / f"checkpoint-{global_step}" / "index"
    indexer.index(index_path, train_dataset["text"], gpus=True)

    sampling_data_builder = TripleSamplingData(train_dataset["label"], **sampling_kwargs)
    data_collator_for_triples = DataCollatorForTriples(tokenizer)
    if do_searched_sampling:
        sampling_input_columns = ["match_pids", "match_scores", "label"]
        searcher = ColBERTSearcher(index_path, encoder, searcher_config)
    else:
        sampling_input_columns = ["label"]

    epoch, global_step = 0, 0
    while epoch < num_train_epochs:
        print(f"\n\n## EPOCH {epoch}\n")

        sampling_dataset = train_dataset
        if do_searched_sampling:
            print("Search dataset")
            searcher.index_path = index_path
            sampling_dataset.set_format(None)  # crucial to avoid leaked semaphore objects in map multiprocessing
            sampling_dataset = sampling_dataset.map(
                run_multiprocessed(searcher.search),
                input_columns="text",
                fn_kwargs={"k": searcher_k},
                batched=True,
                num_proc=torch.cuda.device_count(),
                with_rank=True,
                desc="Searching",
            )

        print("Create Sampling data")
        sampling_dataset.set_format("numpy")
        sampling_dataset = sampling_dataset.map(
            sampling_data_builder,
            input_columns=sampling_input_columns,
            with_indices=True,
            remove_columns=sampling_input_columns,
        )

        print("Tokenize")
        sampling_dataset = sampling_dataset.map(
            lambda batch: tokenizer(batch, truncation=True),
            input_columns="text",
            remove_columns="text",
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
        trainer.train(resume_from_checkpoint=(epoch > 0))  # don't resume in the first epoch

        global_step = trainer.state.global_step
        epoch = trainer.state.epoch
        tokenizer.save_pretrained(output_dir / f"checkpoint-{global_step}")

        if subsample_train is not False:
            print("Reshuffle train dataset subsample")
            train_dataset = train_dataset_.shuffle().take(subsample_train)

        print("Rebuild index")
        index_path = output_dir / f"checkpoint-{global_step}" / "index"
        model.__class__ = ColBERTModel  # cast to ColBERTModel
        indexer.index(index_path, train_dataset["text"], gpus=True)

    # print("Save latest model")
    # trainer.save_model(output_dir / "latest")
    # tokenizer.tokenizer.save_pretrained(output_dir / f"latest")

    # print("Build latest index")
    # # for indexing/searching cast model to basic ColBERTModel
    # model.__class__ = ColBERTModel
    # indexer.index(output_dir / f"latest" / "index", train_dataset["text"], gpus=True)


if __name__ == "__main__":
    from datasets import load_dataset

    model_name_or_path = "bert-base-uncased"
    dataset_name = "imdb"
    model_config = ColBERTConfig.from_pretrained(model_name_or_path, compression_dim=128)

    train_dataset = load_dataset(dataset_name, split="train")
    # train_dataset = train_dataset.rename_column("fine_label", "label")
    training_args = TrainingArguments(
        output_dir=Path("training") / dataset_name / Path(model_name_or_path).name,
        save_total_limit=1,
        save_strategy="epoch",
        save_steps=1,
        num_train_epochs=1,
        # per_device_train_batch_size=16,
        auto_find_batch_size=True,
        optim="adamw_torch",
    )

    train(
        train_dataset=train_dataset,
        model_name_or_path=model_name_or_path,
        output_dir=f"indexes/finetuning/{model_name_or_path}",
        model_config=model_config,
        training_args=training_args,
    )
