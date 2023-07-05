import itertools
from collections.abc import Iterable, Iterator

import numpy as np
import torch
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from gensim.models import FastText, Word2Vec
from gensim.utils import simple_preprocess

from ettcl.encoding.encoder import MultiVectorEncoder, TEncoder
from ettcl.logging.tqdm import tqdm
import torch.nn.functional as F


def preprocess(text: str, max_length: int = 512, **kwargs) -> list[str]:
    return {"tokens": simple_preprocess(text, **kwargs)[:max_length]}


def token_vectors(tokens: list[str], model: FastText, normalize: bool = True) -> np.ndarray:
    vecs = model.wv[tokens]
    if normalize:
        vecs = vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)
    return {"vecs": vecs, "length": vecs.shape[0]}


class Word2VecEncoder(MultiVectorEncoder):
    def __init__(self, model: FastText | Word2Vec, normalize_embeddings: bool = True, **preprocess_kwargs) -> None:
        super().__init__()
        self.model = model
        self.preprocess_kwargs = preprocess_kwargs
        self.normalize = normalize_embeddings
        self.use_gpu = False

    @property
    def embedding_dim(self) -> int:
        return self.model.vector_size

    def cuda(self: TEncoder, device: int | None = None) -> TEncoder:
        self.use_gpu = True
        return self

    def cpu(self: TEncoder) -> TEncoder:
        self.use_gpu = False
        return self

    def encode_passages(
        self,
        passages: list[str],
        *,
        to_cpu: bool = True,
        progress_bar: bool = True,
        keepdims: bool = False,
        return_dict: bool = False,
        **unused_kwargs,
    ) -> tuple[torch.FloatTensor | list[torch.FloatTensor], list[int]]:
        assert len(passages) > 0, "No passages provided"
        if not progress_bar:
            disable_progress_bar()

        use_gpu = self.use_gpu and not to_cpu

        corpus = Dataset.from_dict({"text": passages})

        corpus = corpus.map(
            preprocess,
            input_columns="text",
            fn_kwargs=self.preprocess_kwargs,
            desc="Preprocessing",
        )

        corpus = corpus.map(
            token_vectors,
            input_columns="tokens",
            fn_kwargs={"model": self.model, "normalize": self.normalize},
            desc="Encoding",
        )

        doc_lengths = corpus["length"]

        corpus.set_format("torch")

        if keepdims:
            embeddings = [t.half() for t in corpus["vecs"]]
            embeddings = list(map(lambda t: t.cuda(), embeddings)) if use_gpu else embeddings

        else:
            embeddings = torch.cat(corpus["vecs"]).half()
            embeddings = embeddings.cuda() if use_gpu else embeddings

        if not return_dict:
            return embeddings, doc_lengths

        return {"embeddings": embeddings, "length": doc_lengths}

    def encode_queries(
        self,
        queries: list[str],
        *,
        to_cpu: bool = False,
        progress_bar: bool = True,
        return_dict: bool = False,
        **unused_kwargs,
    ) -> list[torch.FloatTensor]:
        assert len(queries) > 0, "No queries provided"
        embeddings, _ = self.encode_passages(queries, to_cpu=to_cpu, keepdims=True, progress_bar=progress_bar)

        if not return_dict:
            return embeddings

        return {"embeddings": embeddings}

