from datasets import load_dataset, DatasetDict
from itertools import chain
from collections import Counter


def Reuters(multilabel: bool = True):
    train_dataset = load_dataset('reuters21578', 'ModApte', split='train')
    train_dataset = train_dataset.select_columns(['topics', 'text'])
    test_dataset = load_dataset('reuters21578', 'ModApte', split='test')
    test_dataset = test_dataset.select_columns(['topics', 'text'])

    c = Counter(chain(*map(lambda feat: feat['topics'], train_dataset)))
    c = {k: v for k, v in c.items() if v >= 10}

    train_dataset = train_dataset.map(
        lambda topics: {"topics": [t for t in topics if t in c.keys()]},
        input_columns="topics"
    )

    test_dataset = test_dataset.map(
        lambda topics: {"topics": [t for t in topics if t in c.keys()]},
        input_columns="topics"
    )

    label2id = {k: i for i, k in enumerate(c.keys())}
    # id2label = {i: k for k, i in label2id.items()}

    train_dataset = train_dataset.map(lambda topics: {"labels": list({label2id[t] for t in topics})}, input_columns='topics')
    test_dataset = test_dataset.map(lambda topics: {"labels": list({label2id[t] for t in topics})}, input_columns='topics')

    train_dataset = train_dataset.filter(len, input_columns="labels")
    test_dataset = test_dataset.filter(len, input_columns="labels")

    if not multilabel:
        train_dataset = train_dataset.filter(lambda labels: len(labels) == 1, input_columns="labels")
        train_dataset = train_dataset.map(
            lambda labels, topics: {'label': labels[0], 'topic': topics[0]},
            input_columns=['labels', 'topics'],
            remove_columns=['labels', 'topics'],
        )

        test_dataset = test_dataset.filter(lambda labels: len(labels) == 1, input_columns="labels")
        test_dataset = test_dataset.map(
            lambda labels, topics: {'label': labels[0], 'topic': topics[0]},
            input_columns=['labels', 'topics'],
            remove_columns=['labels', 'topics'],
        )

    return DatasetDict({"train": train_dataset, "test": test_dataset})


if __name__ == "__main__":
    dataset = Reuters(multilabel=False)
    dataset.save_to_disk("data/ReutersMCC")
