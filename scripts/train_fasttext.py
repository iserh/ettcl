#!/usr/bin/env python
from pathlib import Path

from datasets import load_from_disk, Split
from gensim.models import FastText
from gensim.utils import simple_preprocess

from ettcl.logging import configure_logger
import os

configure_logger("INFO")

dataset_path = "/home/IAIS/hiser/data/trec-6"
dataset_name = os.path.basename(dataset_path)
text_column = "text"
label_column = "label"

dataset = load_from_disk(dataset_path)[Split.TRAIN]


def preprocess(text: str, max_length: int = 512, **kwargs) -> list[str]:
    return {"tokens": simple_preprocess(text, **kwargs)[:max_length]}


dataset = dataset.map(
    preprocess,
    input_columns="text",
    num_proc=8,
)

model = FastText(vector_size=128, window=5, min_count=5, word_ngrams=1)
model.build_vocab(corpus_iterable=dataset["tokens"])
total_words = model.corpus_total_words
total_examples = model.corpus_count
print(f"{total_examples=}")
print(f"{total_words=}")

model.train(corpus_iterable=dataset["tokens"], total_words=total_words, total_examples=total_examples, epochs=10)

Path(f"models/{dataset_name}").mkdir(exist_ok=True, parents=True)
model.save(f"models/{dataset_name}/fasttext")
