from datasets import DatasetDict, load_dataset

from ettcl.data.utils import train_split


def IMDB():
    train_dataset = load_dataset("imdb", split="train")
    test_dataset = load_dataset("imdb", split="test")
    train_dataset, val_dataset = train_split(train_dataset, size=0.1, stratify="label")

    return DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})


if __name__ == "__main__":
    dataset = IMDB()
    dataset.save_to_disk("~/data/imdb")
