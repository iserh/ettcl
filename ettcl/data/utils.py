from collections import Counter
from itertools import chain

from datasets import Dataset


def count_labels(dataset: Dataset, label_column: str, multilabel: bool = False) -> Counter:
    if multilabel:
        return Counter(chain(*map(lambda feat: feat[label_column], dataset)))
    else:
        return Counter(map(lambda feat: feat[label_column], dataset))
