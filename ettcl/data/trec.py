from datasets import DatasetDict, load_dataset

from ettcl.data.utils import train_split
from ettcl.utils import seed_everything


def TREC(label_column: str = "coarse_label"):
    seed_everything(12345)

    dataset = load_dataset("trec")

    dataset = dataset.rename_column(label_column, "label")
    dataset = dataset.select_columns(["text", "label"])

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_dataset, val_dataset = train_split(train_dataset, size=0.1, stratify="label")

    return DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})


if __name__ == "__main__":
    import os

    data_path = os.path.expanduser("~/data")

    dataset = TREC("coarse_label")
    dataset.save_to_disk(os.path.join(data_path, "trec-6"))

    dataset = TREC("fine_label")
    dataset.save_to_disk(os.path.join(data_path, "trec-50"))
