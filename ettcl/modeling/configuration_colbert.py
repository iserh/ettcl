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
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs) -> PretrainedConfig:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        cls.model_type = config.model_type
        return super(ColBERTConfig, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
