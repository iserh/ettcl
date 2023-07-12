from datasets import load_dataset, DatasetDict
from itertools import chain
from collections import Counter


def Reuters(multilabel: bool = True):
    reuters_train = load_dataset('reuters21578', 'ModApte', split='train')
    reuters_train = reuters_train.select_columns(['topics', 'text'])
    reuters_test = load_dataset('reuters21578', 'ModApte', split='test')
    reuters_test = reuters_test.select_columns(['topics', 'text'])

    c = Counter(chain(*map(lambda feat: feat['topics'], reuters_train)))
    c = {k: v for k, v in c.items() if v >= 10}

    reuters_train = reuters_train.map(
        lambda topics: {"topics": [t for t in topics if t in c.keys()]},
        input_columns="topics"
    )

    reuters_test = reuters_test.map(
        lambda topics: {"topics": [t for t in topics if t in c.keys()]},
        input_columns="topics"
    )

    label2id = {k: i for i, k in enumerate(c.keys())}
    # id2label = {i: k for k, i in label2id.items()}

    reuters_train = reuters_train.map(lambda topics: {"labels": [label2id[t] for t in topics]}, input_columns='topics')
    reuters_test = reuters_test.map(lambda topics: {"labels": [label2id[t] for t in topics]}, input_columns='topics')

    reuters_train = reuters_train.filter(len, input_columns="labels")
    reuters_test = reuters_test.filter(len, input_columns="labels")

    if not multilabel:
        reuters_train = reuters_train.filter(lambda labels: len(labels) == 1, input_columns="labels")
        reuters_train = reuters_train.map(
            lambda labels, topics: {'label': labels[0], 'topic': topics[0]},
            input_columns=['labels', 'topics'],
            remove_columns=['labels', 'topics'],
        )

        reuters_test = reuters_test.filter(lambda labels: len(labels) == 1, input_columns="labels")
        reuters_test = reuters_test.map(
            lambda labels, topics: {'label': labels[0], 'topic': topics[0]},
            input_columns=['labels', 'topics'],
            remove_columns=['labels', 'topics'],
        )

    return DatasetDict({"train": reuters_train, "test": reuters_test})


if __name__ == "__main__":
    dataset = Reuters(multilabel=False)
    dataset.save_to_disk("data/ReutersMCC")
