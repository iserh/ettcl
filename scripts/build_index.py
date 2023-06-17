if __name__ == "__main__":
    from pathlib import Path

    import datasets

    from ettcl.encoding import ColBERTEncoder
    from ettcl.indexing import ColBERTIndexer, IndexerConfig
    from ettcl.modeling import ColBERTModel, ColBERTTokenizer

    retrain_index = True

    dataset = "trec"
    index_dir = "indexes"
    model_path = "bert-base-uncased"

    model = ColBERTModel.from_pretrained(model_path)
    tokenizer = ColBERTTokenizer(model_path)
    encoder = ColBERTEncoder(model, tokenizer)

    train_dataset = datasets.load_dataset(dataset, split="train")

    indexer_config = IndexerConfig(nbits=2)
    index_path = Path(index_dir) / f"{Path(model_path).name}.{indexer_config.nbits}bits"
    indexer = ColBERTIndexer(encoder, indexer_config)
    indexer.index(index_path, train_dataset["text"], gpus=1)
