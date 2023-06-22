import time

import datasets

from ettcl.encoding import ColBERTEncoder
from ettcl.modeling import ColBERTModel, ColBERTTokenizer
from ettcl.searching.colbert_searcher import ColBERTSearcher
from ettcl.utils.multiprocessing import run_multiprocessed

datasets.enable_caching()


def main():
    checkpoint = "training/trec/bert-base-uncased/checkpoint-1023"
    model = ColBERTModel.from_pretrained(checkpoint)
    tokenizer = ColBERTTokenizer(checkpoint)
    encoder = ColBERTEncoder(model, tokenizer)
    searcher = ColBERTSearcher(checkpoint + "/index", encoder)
    dataset = datasets.load_dataset("imdb", split="train")

    start = time.perf_counter()
    dataset = dataset.map(
        run_multiprocessed(searcher.search),
        num_proc=2,
        with_rank=True,
        # searcher.search,
        input_columns="text",
        fn_kwargs={"k": 10},
        batched=True,
        batch_size=5000,
        # load_from_cache_file=False,
    )
    # searcher.search(dataset["text"], k=10, progress_bar=True)
    end = time.perf_counter()

    print(f"[TIME] {end - start:.2f}s")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    main()

    # from datasets.fingerprint import Hasher

    # checkpoint = "training/trec/bert-base-uncased/checkpoint-1023"
    # model = ColBERTModel.from_pretrained(checkpoint)
    # tokenizer = ColBERTTokenizer(checkpoint)
    # encoder = ColBERTEncoder(model, tokenizer)
    # searcher = ColBERTSearcher(checkpoint + "/index", encoder)

    # print("Searcher:", Hasher.hash(searcher))
    # print(Hasher.hash(searcher.ranker.embeddings_strided.codes_strided.strides))
