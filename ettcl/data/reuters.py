from datasets import DatasetDict, load_dataset

from ettcl.data.utils import count_labels, train_split
from datasets import ClassLabel


def Reuters(mlc: bool = True):
    train_dataset = load_dataset("reuters21578", "ModApte", split="train")
    train_dataset = train_dataset.select_columns(["topics", "text"])
    test_dataset = load_dataset("reuters21578", "ModApte", split="test")
    test_dataset = test_dataset.select_columns(["topics", "text"])

    c = count_labels(train_dataset, "topics", multilabel=True)
    c = {k: v for k, v in c.items() if v >= 10}

    train_dataset = train_dataset.map(
        lambda topics: {"topics": [t for t in topics if t in c.keys()]}, input_columns="topics"
    )

    test_dataset = test_dataset.map(
        lambda topics: {"topics": [t for t in topics if t in c.keys()]}, input_columns="topics"
    )

    label2id = {k: i for i, k in enumerate(c.keys())}
    # id2label = {i: k for k, i in label2id.items()}

    train_dataset = train_dataset.map(
        lambda topics: {"labels": list({label2id[t] for t in topics})}, input_columns="topics"
    )
    test_dataset = test_dataset.map(
        lambda topics: {"labels": list({label2id[t] for t in topics})}, input_columns="topics"
    )

    train_dataset = train_dataset.filter(len, input_columns="labels")
    test_dataset = test_dataset.filter(len, input_columns="labels")

    if not mlc:
        train_dataset = train_dataset.filter(lambda labels: len(labels) == 1, input_columns="labels")
        train_dataset = train_dataset.map(
            lambda labels, topics: {"label": labels[0], "topic": topics[0]},
            input_columns=["labels", "topics"],
            remove_columns=["labels", "topics"],
        )

        test_dataset = test_dataset.filter(lambda labels: len(labels) == 1, input_columns="labels")
        test_dataset = test_dataset.map(
            lambda labels, topics: {"label": labels[0], "topic": topics[0]},
            input_columns=["labels", "topics"],
            remove_columns=["labels", "topics"],
        )

        c = count_labels(train_dataset, "label", multilabel=False)
        c = {k: v for k, v in c.items() if v >= 2}

        train_dataset = train_dataset.filter(lambda label: label in c.keys(), input_columns="label")
        test_dataset = test_dataset.filter(lambda label: label in c.keys(), input_columns="label")

        new_features = train_dataset.features.copy()
        new_features["label"] = ClassLabel(len(c.keys()))
        train_dataset = train_dataset.cast(new_features)

        new_features = test_dataset.features.copy()
        new_features["label"] = ClassLabel(len(c.keys()))
        test_dataset = test_dataset.cast(new_features)

    train_dataset, val_dataset = train_split(train_dataset, size=0.1, stratify="label" if not mlc else None)

    return DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})


if __name__ == "__main__":
    dataset = Reuters(mlc=True)
    dataset.save_to_disk("~/data/ReutersMLC")

    dataset = Reuters(mlc=False)
    dataset.save_to_disk("~/data/ReutersCLS")
