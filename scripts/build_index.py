#!/usr/bin/env python
import os

import datasets

from ettcl.encoding import ColBERTEncoder
from ettcl.indexing import ColBERTIndexer, ColBERTIndexerConfig
from ettcl.modeling import ColBERTConfig, ColBERTModel, ColBERTTokenizer
from ettcl.utils.utils import catchtime


def main():
    dataset_name = "imdb"
    pretrained_model_name_or_path = "bert-base-uncased"
    save_model = True
    nbits = 2
    index_name = f"{pretrained_model_name_or_path}.{nbits}bits"

    dataset = datasets.load_dataset(dataset_name, split="train")

    model_config = ColBERTConfig.from_pretrained(pretrained_model_name_or_path, compression_dim=128)
    model = ColBERTModel.from_pretrained(pretrained_model_name_or_path, config=model_config)
    tokenizer = ColBERTTokenizer.from_pretrained(pretrained_model_name_or_path, query_maxlen=256, doc_maxlen=256)
    encoder = ColBERTEncoder(model, tokenizer)

    indexer_config = ColBERTIndexerConfig(nbits=nbits)
    indexer = ColBERTIndexer(encoder, indexer_config)
    index_path = os.path.join("indexes", dataset_name, index_name)

    with catchtime(desc="Indexing"):
        indexer.index(index_path, dataset["text"], gpus=True)

    if save_model:
        model.save_pretrained(index_path)
        tokenizer.save_pretrained(index_path)


if __name__ == "__main__":
    with catchtime(desc="Main"):
        main()
