from datasets import DatasetDict, load_dataset


def AmazonCat_13K() -> DatasetDict:
    dataset = load_dataset("ettcl/data/xcr.py", name="AmazonCat-13K")

    dataset = dataset.map(
        lambda title, text: {"titletext": title + " \n" + text},
        input_columns=["title", "text"],
    )
    dataset = dataset.remove_columns(["title", "text"])
    dataset = dataset.rename_column("titletext", "text")

    return dataset


if __name__ == "__main__":
    dataset = AmazonCat_13K()
    dataset.save_to_disk("data/AmazonCat-13K")
