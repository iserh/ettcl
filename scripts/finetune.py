#!/usr/bin/env python
from pathlib import Path

from colbert.infra.config.config import ColBERTConfig
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from ettcl.core.triple_sampling import DataCollatorForTriples, TripleSampler, create_triples_ranked
from ettcl.encoding import ColBERTEncoder, EncoderFactory
from ettcl.indexing import ColBERTIndexer, IndexerConfig
from ettcl.modeling import ColbertConfig, ColbertForReranking
from ettcl.searching import ColBERTSearcher, SearcherConfig


def train(
    train_dataset: Dataset,
    model_name_or_path: str,
    index_path: str,
    colbert_config: ColBERTConfig,
    training_args: TrainingArguments,
    triples_generator: callable,
    test_dataset: Dataset | None = None,
) -> None:
    output_dir = Path(training_args.output_dir)
    index_path = Path(index_path)
    num_train_epochs = training_args.num_train_epochs

    model_config = ColbertConfig.from_pretrained(
        model_name_or_path, compression_dim=colbert_config.dim
    )
    model = ColbertForReranking.from_pretrained(model_name_or_path, config=model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.save_pretrained(output_dir / "latest")

    train_dataset = train_dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True), batched=True
    )
    data_collator = DataCollatorForTriples(tokenizer)

    searching_args = SearcherConfig(gpus=1, ncells=32)
    indexing_args = IndexerConfig(gpus=1, nbits=2, dim=colbert_config.dim)
    encoder_factory = EncoderFactory(
        ColBERTEncoder, checkpoint=model_name_or_path, config=colbert_config
    )
    indexer = ColBERTIndexer(encoder_factory=encoder_factory, config=indexing_args)

    for epoch in range(1, num_train_epochs + 1):
        print(f"\n\n## EPOCH {epoch}\n")
        print("Rebuild index")
        indexer.index(index_path / f"epoch={epoch}", train_dataset["text"])
        print("Create Searcher")
        searcher = ColBERTSearcher(index_path / f"epoch={epoch}", encoder_factory, searching_args)
        print("Create triples dataset")
        train_dataset_with_triples = triples_generator(
            train_dataset, searcher, k=256, sampling_mode="scores", nway=2
        )
        print(f"# NumExamples={len(train_dataset_with_triples)}")
        train_dataset_with_triples = train_dataset_with_triples.map(
            lambda batch: tokenizer(batch["text"], truncation=True), batched=True
        )
        train_dataset_with_triples = train_dataset_with_triples.select_columns(
            ["input_ids", "token_type_ids", "attention_mask", "triple"]
        )

        print("Create Trainer")
        sampler = TripleSampler(train_dataset_with_triples)
        training_args.num_train_epochs = epoch
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=sampler,
            data_collator=data_collator,
        )
        print("Train")
        trainer.train(resume_from_checkpoint=(epoch > 1))

    trainer.save_model(output_dir / "latest")


if __name__ == "__main__":
    from datasets import load_dataset

    model_name_or_path = "bert-base-uncased"
    # colbert_config = ColBERTConfig.load_from_checkpoint(model_name_or_path)
    colbert_config = ColBERTConfig(dim=768)

    train_dataset = load_dataset("trec", split="train")
    train_dataset = train_dataset.rename_column("fine_label", "label")
    training_args = TrainingArguments(
        output_dir=Path("training") / Path(model_name_or_path).name,
        save_total_limit=1,
        save_steps=50,
        num_train_epochs=3,
        # per_device_train_batch_size=16,
        auto_find_batch_size=True,
        optim="adamw_torch",
    )

    train(
        train_dataset=train_dataset,
        model_name_or_path=model_name_or_path,
        index_path=f"indexes/finetuning/{model_name_or_path}",
        colbert_config=colbert_config,
        training_args=training_args,
        triples_generator=create_triples_ranked,
    )
