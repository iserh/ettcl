from __future__ import annotations

import os
from enum import Enum

import wrapt
from transformers import AutoTokenizer, PreTrainedTokenizerBase, logging
from transformers.tokenization_utils import (
    BatchEncoding,
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)

logger = logging.get_logger(__name__)

COLBERT_TOKENIZER_CONFIG_FILE = "colbert_tokenizer_config.json"


class TokenizerMode(str, Enum):
    query = "query"
    doc = "doc"


class ColBERTTokenizer(wrapt.ObjectProxy):
    special_tokens_pretty = {
        "query": "[Q]",
        "doc": "[D]",
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        query_token: str | None = "[unused0]",
        doc_token: str | None = "[unused1]",
        query_maxlen: int = 512,
        doc_maxlen: int = 512,
        query_augmentation: bool = False,
        attend_to_mask_tokens: bool = False,
        add_special_tokens: bool = True,
        *inputs,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer)
        self.init_kwargs.update(
            dict(
                query_token=query_token,
                doc_token=doc_token,
                query_maxlen=query_maxlen,
                doc_maxlen=doc_maxlen,
                query_augmentation=query_augmentation,
                attend_to_mask_tokens=attend_to_mask_tokens,
            )
        )

        self.query_token = query_token
        self.doc_token = doc_token
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.query_augmentation = query_augmentation
        self.attend_to_mask_tokens = attend_to_mask_tokens
        self.def_add_special_tokens = add_special_tokens

        if self.attend_to_mask_tokens and not self.query_augmentation:
            logger.warning("With `query_augmentation` disabled, `attend_to_mask_tokens` (set to True) will be ignored.")

        special_tokens = [token for token in [self.query_token, self.doc_token] if token is not None]
        num_add_tokens = self.__wrapped__.add_special_tokens({"additional_special_tokens": special_tokens})
        if num_add_tokens > 0:
            logger.warning(
                f"Added special tokens {special_tokens} to the tokenizer "
                "that were not in vocab. You should ensure to call `model.resize_token_embeddings` "
                "to match the models embedding matrix."
            )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, *init_inputs, **kwargs
    ) -> ColBERTTokenizer | PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        return cls(tokenizer, *init_inputs, **kwargs)

    def tokenize(
        self,
        text: str | list[str],
        mode: TokenizerMode = "doc",
        add_special_tokens: bool | None = None,
        pretty: bool = True,
        **kwargs,
    ) -> list[list[str]] | list[str]:
        if add_special_tokens is None:
            add_special_tokens = self.def_add_special_tokens

        special_token = self.doc_token if mode == "doc" else self.query_token
        special_token = special_token or ""

        batched = not isinstance(text, str)
        text = [text] if not batched else text

        tokens = [
            self.__wrapped__.tokenize(special_token + s, add_special_tokens=add_special_tokens, **kwargs) for s in text
        ]

        if pretty:
            special_token_pretty = self.special_tokens_pretty[mode]
            for t in tokens:
                t[1] = special_token_pretty

        return tokens if batched else tokens[0]

    def pad(
        self,
        encoded_inputs,
        mode: TokenizerMode = "doc",
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = None,
        return_tensors: str | TensorType | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        if max_length is None and padding == "max_length":
            max_length = self.doc_maxlen if mode == "doc" else self.query_maxlen

        return self.__wrapped__.pad(
            encoded_inputs=encoded_inputs,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
            verbose=verbose,
        )

    def __call__(
        self,
        text: str | list[str],
        mode: TokenizerMode = "doc",
        add_special_tokens: bool | None = None,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy = True,
        return_tensors: str | TensorType | None = None,
        max_length: str | None = None,
        return_token_type_ids: bool | None = False,
        **kwargs,
    ) -> BatchEncoding:
        if isinstance(text, str):
            text = [text]

        if add_special_tokens is None:
            add_special_tokens = self.def_add_special_tokens

        # add special token
        special_token = self.doc_token if mode == "doc" else self.query_token
        if special_token is not None:
            text = [special_token + s for s in text]

        if max_length is None:
            max_length = self.doc_maxlen if mode == "doc" else self.query_maxlen

        encoding = self.__wrapped__(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            **kwargs,
        )

        input_ids = encoding["input_ids"]
        if self.query_augmentation:
            encoding["input_ids"][input_ids == self.tokenizer.pad_token_id] = self.tokenizer.mask_token_id
            if self.attend_to_mask_tokens:
                encoding["attention_mask"][input_ids == self.tokenizer.mask_token_id]

        return encoding

    # for pickling (ObjectProxy does not support this out of the box)
    def __reduce_ex__(self, protocol_version):
        """Pickle reduce method"""
        return (self._unpickle, (self.__wrapped__, self.init_kwargs))

    @classmethod
    def _unpickle(cls, tokenizer: PreTrainedTokenizerBase, init_kwargs: dict) -> ColBERTTokenizer:
        return cls(tokenizer, **init_kwargs)
