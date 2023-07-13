from datasets import DatasetDict, load_dataset
from ettcl.data.utils import count_labels


def AmazonCat_13K() -> DatasetDict:
    dataset = load_dataset("ettcl/data/xcr.py", name="AmazonCat-13K")

    dataset = dataset.map(
        lambda title, text: {"titletext": title + " \n" + text},
        input_columns=["title", "text"],
    )
    dataset = dataset.remove_columns(["title", "text"])
    dataset = dataset.rename_column("titletext", "text")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    c = count_labels(train_dataset, 'labels', multilabel=True)
    c = {k: v for k, v in c.items() if v >= 20}

    train_dataset = train_dataset.map(
        lambda labels: {"labels": [t for t in labels if t in c.keys()]},
        input_columns="labels"
    )

    test_dataset = test_dataset.map(
        lambda labels: {"labels": [t for t in labels if t in c.keys()]},
        input_columns="labels"
    )

    label2id = {k: i for i, k in enumerate(c.keys())}
    # id2label = {i: k for k, i in label2id.items()}

    train_dataset = train_dataset.map(lambda labels: {"labels": list({label2id[t] for t in labels})}, input_columns='labels')
    test_dataset = test_dataset.map(lambda labels: {"labels": list({label2id[t] for t in labels})}, input_columns='labels')

    train_dataset = train_dataset.filter(len, input_columns="labels")
    test_dataset = test_dataset.filter(len, input_columns="labels")

    return DatasetDict({"train": train_dataset, "test": test_dataset})


if __name__ == "__main__":
    dataset = AmazonCat_13K()
    dataset.save_to_disk("data/AmazonCat-13K")
