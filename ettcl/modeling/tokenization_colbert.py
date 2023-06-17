from enum import Enum

from transformers import AutoTokenizer, logging
from transformers.tokenization_utils import (
    BatchEncoding,
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)

from ettcl.modeling.configuration_colbert import ColBERTTokenizerConfig

logger = logging.get_logger(__name__)


class TokenizerMode(str, Enum):
    query = "query"
    doc = "doc"


class ColBERTTokenizer:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        config: ColBERTTokenizerConfig = ColBERTTokenizerConfig(),
        *inputs,
        **kwargs,
    ) -> None:
        self.config = config
        self.query_token = config.query_token
        self.doc_token = config.doc_token
        self.query_maxlen = config.query_maxlen
        self.doc_maxlen = config.doc_maxlen
        self.query_augmentation = config.query_augmentation
        self.attend_to_mask_tokens = config.attend_to_mask_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, **kwargs
        )

        special_tokens = [
            token for token in [self.query_token, self.doc_token] if token is not None
        ]
        num_add_tokens = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        if num_add_tokens > 0:
            logger.warning(
                f"Added special tokens {special_tokens} to the tokenizer "
                "that were not in vocab. You should ensure to call `model.resize_token_embeddings` "
                "to match the models embedding matrix."
            )

    def tokenize(
        self,
        text: str | list[str],
        mode: TokenizerMode = "doc",
        add_special_tokens: bool = True,
        **kwargs,
    ) -> list[list[str]] | list[str]:
        special_token = self.doc_token if mode == "doc" else self.query_token
        special_token = special_token or ""

        if isinstance(text, str):
            return self.tokenizer.tokenize(
                special_token + text, add_special_tokens=add_special_tokens, **kwargs
            )
        else:
            return [
                self.tokenizer.tokenize(
                    special_token + s, add_special_tokens=add_special_tokens, **kwargs
                )
                for s in text
            ]

    def __call__(
        self,
        text: str | list[str],
        mode: TokenizerMode = "doc",
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy = True,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchEncoding:
        if isinstance(text, str):
            text = [text]

        # add special token
        special_token = self.doc_token if mode == "doc" else self.query_token
        if special_token is not None:
            text = [special_token + s for s in text]

        encoding = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=self.query_maxlen,
            return_tensors=return_tensors,
            **kwargs,
        )

        input_ids = encoding["input_ids"]
        if self.query_augmentation:
            encoding["input_ids"][
                input_ids == self.tokenizer.pad_token_id
            ] = self.tokenizer.mask_token_id
            if self.attend_to_mask_tokens:
                encoding["attention_mask"][input_ids == self.tokenizer.mask_token_id]

        return encoding
