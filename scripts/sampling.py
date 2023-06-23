#!/usr/bin/env python
import datasets
import torch

from ettcl.core.triple_sampling import TripleSampler
from ettcl.encoding import ColBERTEncoder
from ettcl.modeling import ColBERTModel, ColBERTTokenizer
from ettcl.searching import ColBERTSearcher, ColBERTSearcherConfig
from ettcl.utils.multiprocessing import run_multiprocessed
from ettcl.utils.utils import catchtime


def main():
    dataset_name = "imdb"
    label_column = "label"
    pretrained_model_name_or_path = "bert-base-uncased"
    index_path = "indexes/imdb/bert-base-uncased.2bits"
    searcher_k = 256
    ncells = 1
    centroid_score_threshold = 0.8
    sampling_mode = "scores"
    nway = 2

    dataset = datasets.load_dataset(dataset_name, split="train")

    model = ColBERTModel.from_pretrained(pretrained_model_name_or_path)
    tokenizer = ColBERTTokenizer.from_pretrained(pretrained_model_name_or_path)
    encoder = ColBERTEncoder(model, tokenizer)

    searcher_config = ColBERTSearcherConfig(ncells, centroid_score_threshold)
    searcher = ColBERTSearcher(index_path, encoder, searcher_config)
    sampler = TripleSampler(dataset[label_column], sampling_mode, nway, searcher_k)

    with catchtime(desc="Searching"):
        dataset.set_format(None)  # crucial to avoid leaked semaphore objects in map multiprocessing
        dataset = dataset.map(
            run_multiprocessed(searcher.search),
            input_columns="text",
            fn_kwargs={"k": searcher_k},
            batched=True,
            num_proc=torch.cuda.device_count(),
            with_rank=True,
            desc="Searching",
        )

    with catchtime(desc="Sampling"):
        dataset.set_format("numpy")
        dataset = dataset.map(
            sampler.sample_triples,
            input_columns=["match_pids", "match_scores", label_column],
            desc="Sampling",
        )


if __name__ == "__main__":
    with catchtime(desc="Main"):
        main()
