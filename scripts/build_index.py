if __name__ == "__main__":
    import datasets
    from ettcl.encoding import ColBERTEncoder, EncoderFactory
    from ettcl.indexing import ColBERTIndexer, IndexingArguments

    retrain_index = True

    dataset = "trec"
    label_column = "coarse_label"
    checkpoint = "models/colbertv2.0"
    embedding_dim = 128  # default for colbertv2.0
    encoder_factory = EncoderFactory(ColBERTEncoder, checkpoint=checkpoint)
    index_path = "indexes/colbertv2.0.2bits"

    train_dataset = datasets.load_dataset(dataset, split="train")
    train_dataset.set_format("pt", columns=[label_column])
    test_dataset = datasets.load_dataset(dataset, split="test")
    test_dataset.set_format("pt", columns=[label_column])

    if retrain_index:
        indexing_args = IndexingArguments(dim=embedding_dim, nranks=2, nbits=2)
        indexer = ColBERTIndexer(encoder_factory, indexing_args)
        indexer.index(index_path, train_dataset["text"][:512])
