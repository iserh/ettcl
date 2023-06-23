#!/usr/bin/env python
import datasets
import evaluate
import torch

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

    train_dataset = datasets.load_dataset(dataset_name, split="train")
    test_dataset = datasets.load_dataset(dataset_name, split="test")

    model = ColBERTModel.from_pretrained(pretrained_model_name_or_path)
    tokenizer = ColBERTTokenizer.from_pretrained(pretrained_model_name_or_path)
    encoder = ColBERTEncoder(model, tokenizer)

    searcher_config = ColBERTSearcherConfig(ncells, centroid_score_threshold)
    searcher = ColBERTSearcher(index_path, encoder, searcher_config)
    accuracy_metric = evaluate.load("accuracy")

    with catchtime(desc="Searching"):
        test_dataset.set_format(None)  # crucial to avoid leaked semaphore objects in map multiprocessing
        test_dataset = test_dataset.map(
            run_multiprocessed(searcher.search),
            input_columns="text",
            fn_kwargs={"k": searcher_k},
            batched=True,
            num_proc=torch.cuda.device_count(),
            with_rank=True,
            desc="Searching",
        )

    train_dataset.set_format("pt")
    test_dataset.set_format("pt")

    match_pids = test_dataset["match_pids"]
    match_labels = train_dataset[label_column][match_pids.tolist()]
    y_pred = torch.mode(match_labels)[0]

    acc = accuracy_metric.compute(predictions=y_pred, references=test_dataset[label_column])["accuracy"]
    print(f"Accuracy: {100*acc:.3f}")


if __name__ == "__main__":
    with catchtime(desc="Main"):
        main()
