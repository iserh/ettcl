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
    dataset_name = "trec"
    label_column = "label"
    pretrained_model_name_or_path = "training/trec/bert-base-uncased/checkpoint-345"
    index_path = "training/trec/bert-base-uncased/checkpoint-345/index"
    searcher_k = 32
    ncells = 32
    centroid_score_threshold = None

    train_dataset = datasets.load_dataset(dataset_name, split="train")
    test_dataset = datasets.load_dataset(dataset_name, split="test")

    model = ColBERTModel.from_pretrained(pretrained_model_name_or_path)
    tokenizer = ColBERTTokenizer.from_pretrained(pretrained_model_name_or_path)
    encoder = ColBERTEncoder(model, tokenizer)

    searcher_config = ColBERTSearcherConfig(ncells, centroid_score_threshold)
    searcher = ColBERTSearcher(index_path, encoder, searcher_config)
    clf_metrics = evaluate.combine(["accuracy", "f1"])

    with catchtime(desc="Searching"):
        test_dataset.set_format(None)  # crucial to avoid leaked semaphore objects in map multiprocessing
        test_dataset = test_dataset.map(
            # run_multiprocessed(searcher.search),
            searcher.search,
            input_columns="text",
            fn_kwargs={"k": searcher_k},
            batched=True,
            # num_proc=torch.cuda.device_count(),
            # with_rank=True,
            desc="Searching",
        )

    # train_dataset.set_format("pt")
    # test_dataset.set_format("pt")

    # match_pids = test_dataset["match_pids"]
    # match_labels = train_dataset[label_column][match_pids.tolist()]
    # y_pred = torch.mode(match_labels)[0]

    # metrics = clf_metrics.compute(predictions=y_pred, references=test_dataset[label_column])
    # print(metrics)


if __name__ == "__main__":
    with catchtime(desc="Main"):
        main()
