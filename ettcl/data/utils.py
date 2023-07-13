from collections import Counter
from itertools import chain

from datasets import Dataset


def count_labels(dataset: Dataset, label_column: str, multilabel: bool = False) -> Counter:
    if multilabel:
        return Counter(chain(*map(lambda feat: feat[label_column], dataset)))
    else:
        return Counter(map(lambda feat: feat[label_column], dataset))


def train_split(dataset: Dataset, size: int | float, stratify: str | None = None) -> tuple[Dataset, Dataset]:
    split = dataset.train_test_split(size, stratify_by_column=stratify)
    train_dataset = split["train"]
    val_dataset = split["test"]
    return train_dataset, val_dataset
