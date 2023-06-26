import json
import os
from dataclasses import dataclass
from enum import Enum

from transformers import AutoTokenizer, logging
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


@dataclass
class ColBERTTokenizerConfig:
    query_token: str | None = "[unused0]"
    doc_token: str | None = "[unused1]"
    query_augmentation: bool = False
    attend_to_mask_tokens: bool = False
    query_maxlen: int = 512
    doc_maxlen: int = 512

    def __post_init__(self) -> None:
        if self.attend_to_mask_tokens and not self.query_augmentation:
            logger.warning("With `query_augmentation` disabled, `attend_to_mask_tokens` (set to True) will be ignored.")


class ColBERTTokenizer:
    special_tokens_pretty = {
        "query": "[Q]",
        "doc": "[D]",
    }

    def __init__(
        self,
        pretrained_model_name_or_path: str | os.PathLike,
        query_token: str | None = "[unused0]",
        doc_token: str | None = "[unused1]",
        query_augmentation: bool = False,
        attend_to_mask_tokens: bool = False,
        query_maxlen: int = 512,
        doc_maxlen: int = 512,
        *args,
        **kwargs,
    ) -> None:
        self.query_token = query_token
        self.doc_token = doc_token
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.query_augmentation = query_augmentation
        self.attend_to_mask_tokens = attend_to_mask_tokens

        if self.attend_to_mask_tokens and not self.query_augmentation:
            logger.warning("With `query_augmentation` disabled, `attend_to_mask_tokens` (set to True) will be ignored.")

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        special_tokens = [token for token in [self.query_token, self.doc_token] if token is not None]
        num_add_tokens = self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        if num_add_tokens > 0:
            logger.warning(
                f"Added special tokens {special_tokens} to the tokenizer "
                "that were not in vocab. You should ensure to call `model.resize_token_embeddings` "
                "to match the models embedding matrix."
            )

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> tuple[str]:
        files_saved = self.tokenizer.save_pretrained(
            save_directory, legacy_format, filename_prefix, push_to_hub, **kwargs
        )

        colbert_tokenizer_config = {
            "query_token": self.query_token,
            "doc_token": self.doc_token,
            "query_maxlen": self.query_maxlen,
            "doc_maxlen": self.doc_maxlen,
            "query_augmentation": self.query_augmentation,
            "attend_to_mask_tokens": self.attend_to_mask_tokens,
        }

        colbert_tokenizer_config_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + COLBERT_TOKENIZER_CONFIG_FILE
        )

        with open(colbert_tokenizer_config_file, "w", encoding="utf-8") as f:
            out_str = json.dumps(colbert_tokenizer_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(out_str)
        logger.info(f"colbert tokenizer config file saved in {colbert_tokenizer_config_file}")

        return (colbert_tokenizer_config_file, *files_saved)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, *init_inputs, **kwargs
    ) -> "ColBERTTokenizer":
        colbert_tokenizer_config_file = os.path.join(pretrained_model_name_or_path, COLBERT_TOKENIZER_CONFIG_FILE)
        if os.path.exists(colbert_tokenizer_config_file):
            with open(colbert_tokenizer_config_file, "r", encoding="utf-8") as f:
                colbert_tokenizer_config = json.load(f)
        else:
            colbert_tokenizer_config = {}
        colbert_tokenizer_config.update(kwargs)

        return cls(pretrained_model_name_or_path, *init_inputs, **colbert_tokenizer_config)

    def tokenize(
        self,
        text: str | list[str],
        mode: TokenizerMode = "doc",
        add_special_tokens: bool = True,
        pretty: bool = True,
        **kwargs,
    ) -> list[list[str]] | list[str]:
        special_token = self.doc_token if mode == "doc" else self.query_token
        special_token = special_token or ""

        batched = not isinstance(text, str)
        text = [text] if batched else text

        tokens = [
            self.tokenizer.tokenize(special_token + s, add_special_tokens=add_special_tokens, **kwargs) for s in text
        ]

        if pretty:
            special_token_pretty = self.special_tokens_pretty[mode]
            for t in tokens:
                t[1] == special_token_pretty

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
    ) -> BatchEncoding:
        if max_length is None and padding == "max_length":
            max_length = self.doc_maxlen if mode == "doc" else self.query_maxlen

        return self.tokenizer.pad(
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

        max_length = kwargs.pop("max_length", None)
        if max_length is None:
            max_length = self.doc_maxlen if mode == "doc" else self.query_maxlen

        encoding = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

        input_ids = encoding["input_ids"]
        if self.query_augmentation:
            encoding["input_ids"][input_ids == self.tokenizer.pad_token_id] = self.tokenizer.mask_token_id
            if self.attend_to_mask_tokens:
                encoding["attention_mask"][input_ids == self.tokenizer.mask_token_id]

        return encoding
