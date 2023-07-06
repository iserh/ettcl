from datasets import Dataset, load_dataset
from gensim.models import FastText
from gensim.utils import simple_preprocess

from ettcl.logging import configure_logger

configure_logger("INFO")

dataset_name = "imdb"
text_column = "text"
label_column = "label"

train_dataset = load_dataset(dataset_name, split="train")
train_dataset = train_dataset.rename_columns({text_column: "text", label_column: "label"})


def preprocess(text: str, max_length: int = 512, **kwargs) -> list[str]:
    return {"tokens": simple_preprocess(text, **kwargs)[:max_length]}


corpus = Dataset.from_dict({"text": train_dataset["text"]})

corpus = corpus.map(
    preprocess,
    input_columns="text",
    num_proc=8,
)

model = FastText(vector_size=128, window=5, min_count=5)
model.build_vocab(corpus_iterable=corpus["tokens"])
total_words = model.corpus_total_words
total_examples = model.corpus_count
print(f"{total_examples=}")
print(f"{total_words=}")

model.train(corpus_iterable=corpus["tokens"], total_words=total_words, total_examples=total_examples, epochs=10)

model.save("models/imdb/fasttext")
