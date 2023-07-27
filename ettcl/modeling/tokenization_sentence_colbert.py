from __future__ import annotations

import itertools

import numpy as np
import torch
from transformers.tokenization_utils import BatchEncoding, TensorType

from ettcl.modeling.tokenization_colbert import ColBERTTokenizer, TokenizerMode


class SentenceTokenizer(ColBERTTokenizer):
    """Highly experimental!!"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *inputs,
        add_special_tokens: bool = True,
        query_maxlen: int = 512,
        doc_maxlen: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens,
            query_maxlen=query_maxlen,
            doc_maxlen=doc_maxlen,
            query_token=None,
            doc_token=None,
            query_augmentation=False,
            attend_to_mask_tokens=False,
        )

    def tokenize(self, text: str | list[str], **kwargs) -> list[list[str]] | list[str]:
        return text

    def pad(
        self,
        encoded_inputs,
        mode: TokenizerMode = "doc",
        **kwargs,
    ) -> BatchEncoding:
        keys = list(encoded_inputs[0].keys())
        max_length = self.doc_maxlen if mode == "doc" else self.query_maxlen
        max_num_sentences = np.max([len(feat[keys[0]]) for feat in encoded_inputs])

        batch = {k: [] for k in keys}
        for feat in encoded_inputs:
            for k, v in feat.items():
                p2d = (0, max_length - v.shape[-1], 0, max_num_sentences - v.shape[-2])
                padded = torch.nn.functional.pad(v, p2d, "constant", 0)
                batch[k].append(padded)

        batch = {k: torch.stack(v).view(len(encoded_inputs), max_num_sentences, max_length) for k, v in batch.items()}
        return batch

    def __call__(
        self,
        text: str | list[str],
        return_length: bool = False,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchEncoding:
        assert isinstance(text, list), "Must provide list of sentences"
        if isinstance(text[0], str):
            text = [text]

        num_sentences = list(map(len, text))
        sents_flattened = list(itertools.chain(*text))

        encodings_flatten = super().__call__(sents_flattened, **kwargs)
        keys = set(encodings_flatten.keys())

        if return_length:
            keys = keys | {"length"}

        sent_encodings = {k: [] for k in keys}
        offsets = [0, *np.cumsum(num_sentences)]
        for offset, endpos in zip(offsets[:-1], offsets[1:]):
            encodings = {k: v[offset:endpos] for k, v in encodings_flatten.items()}
            padded = super().pad(encodings, return_tensors="pt", **kwargs)
            for k, v in padded.items():
                sent_encodings[k].append(v)

            if return_length:
                sent_encodings["length"].append(endpos - offset)

        return sent_encodings
