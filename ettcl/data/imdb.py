from datasets import DatasetDict, load_dataset

from ettcl.data.utils import train_split
from ettcl.utils import seed_everything


def IMDB():
    seed_everything(12345)

    train_dataset = load_dataset("imdb", split="train")
    test_dataset = load_dataset("imdb", split="test")
    train_dataset, val_dataset = train_split(train_dataset, size=0.1, stratify="label")

    return DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})


if __name__ == "__main__":
    import os

    data_path = os.path.expanduser("~/data")

    dataset = IMDB()
    dataset.save_to_disk(os.path.join(data_path, "imdb"))
