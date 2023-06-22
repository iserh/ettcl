from dataclasses import dataclass

from colbert.infra.config.base_config import BaseConfig
from colbert.infra.config.core_config import DefaultVal
from colbert.infra.config.settings import IndexingSettings, RunSettings


@dataclass
class _IndexerSettings(BaseConfig, RunSettings, IndexingSettings):
    dim: int = DefaultVal(128)
