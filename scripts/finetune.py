#!/usr/bin/env python
# from multiprocess import set_start_method
from pathlib import Path

from datasets import Dataset
from transformers import Trainer, TrainingArguments

from ettcl.core.triple_sampling import DataCollatorForTriples, TripleSampler, TripleSampleBuilder
from ettcl.encoding import ColBERTEncoder
from ettcl.indexing import ColBERTIndexer
from ettcl.modeling import ColBERTForReranking, ColBERTConfig, ColBERTModel, ColBERTTokenizer
from ettcl.searching import ColBERTSearcher, SearcherConfig
import hashlib
import torch

# try:
#     set_start_method("spawn")
# except:
#     pass


def train(
    train_dataset: Dataset,
    model_name_or_path: str,
    output_dir: str,
    model_config: ColBERTConfig,
    training_args: TrainingArguments,
    test_dataset: Dataset | None = None,
) -> None:
    output_dir = Path(training_args.output_dir)
    num_train_epochs = training_args.num_train_epochs

    model = ColBERTForReranking.from_pretrained(model_name_or_path, config=model_config)
    tokenizer = ColBERTTokenizer(model_name_or_path)

    data_collator = DataCollatorForTriples(tokenizer.tokenizer)

    encoder = ColBERTEncoder(model, tokenizer)
    indexer = ColBERTIndexer(encoder)
    sample_builder = TripleSampleBuilder(train_dataset["label"], sampling_mode="scores")
    searching_args = SearcherConfig(ncells=32)

    print("Build index")
    global_step = 0
    index_path = output_dir / f"checkpoint-{global_step}" / "index"
    # for indexing/searching cast model to basic ColBERTModel
    model.__class__ = ColBERTModel
    # indexer.index(index_path, train_dataset["text"], gpus=True)
    searcher = ColBERTSearcher(index_path, encoder, searching_args)

    for epoch in range(num_train_epochs):
        print(f"\n\n## EPOCH {epoch}\n")

        print("Create triples dataset")
        searcher.index_path = index_path
        train_dataset = train_dataset.map(
            sample_builder.create_samples,
            input_columns=["text", "label"],
            fn_kwargs={"searcher": searcher, "k": 1024, "nway": 32},
            with_rank=True,
            batched=True,
            batch_size=12_500,
            num_proc=torch.cuda.device_count(),
            new_fingerprint=hashlib.md5(str(index_path).encode("utf-8")).hexdigest()
        )

        print("Tokenize")
        train_dataset = train_dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True), batched=True
        )

        print("Create Trainer")
        sampler = TripleSampler(train_dataset.select_columns(["input_ids", "token_type_ids", "attention_mask", "triple"]))
        training_args.num_train_epochs = epoch + 1
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=sampler,
            data_collator=data_collator,
        )


        print("Train")
        # For training cast model to Reanking
        model.__class__ = ColBERTForReranking
        trainer.train(resume_from_checkpoint=(epoch > 0))

        global_step = trainer.state.global_step
        tokenizer.tokenizer.save_pretrained(output_dir / f"checkpoint-{global_step}")

        print("Build index")
        index_path = output_dir / f"checkpoint-{global_step}" / "index"
        # for indexing/searching cast model to basic ColBERTModel
        model.__class__ = ColBERTModel
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
