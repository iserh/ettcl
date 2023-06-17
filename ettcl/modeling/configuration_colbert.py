import os
from dataclasses import dataclass
from typing import Any, Dict, Union

from transformers import AutoConfig, PretrainedConfig, logging
from transformers.configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)


class ColBERTConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compression_dim: int | None = kwargs.pop("compression_dim", None)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> PretrainedConfig:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        cls.model_type = config.model_type
        return super(ColBERTConfig, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)


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
            logger.warning(
                "With `query_augmentation` disabled, `attend_to_mask_tokens` (set to True) will be ignored."
            )

    # TODO: implement a from_pretrained/load Method for this config
