import nltk
import pandas as pd
from datasets import Dataset, DatasetDict
from nltk.corpus import reuters


def Reuters():
    nltk.download("reuters")

    # Extract fileids from the reuters corpus
    fileids = reuters.fileids()

    # Initialize empty lists to store categories and raw text
    labels = []
    text = []

    # Loop through each file id and collect each files categories and raw text
    for file in fileids:
        labels.append(reuters.categories(file))
        text.append(reuters.raw(file))

    # Combine lists into pandas dataframe. reutersDf is the final dataframe.
    reutersDf = pd.DataFrame({"ids": fileids, "labels_text": labels, "text": text})
    reutersDf["split"] = reutersDf.apply(lambda row: row["ids"].split("/")[0], axis=1)

    labels = np.unique(np.concatenate(reutersDf["labels_text"]))
    label2id = {label: i for i, label in enumerate(labels)}
    reutersDf["labels"] = reutersDf.apply(lambda row: [label2id[l] for l in row["labels_text"]], axis=1)

    train_dataset = Dataset.from_pandas(reutersDf[reutersDf["split"] == "training"][["labels_text", "labels", "text"]])
    test_dataset = Dataset.from_pandas(reutersDf[reutersDf["split"] == "test"][["labels_text", "labels", "text"]])
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    return dataset


if __name__ == "__main__":
    dataset = Reuters()
    dataset.save_to_disk("data/Reuters")
