from dataclasses import dataclass
from typing import Literal

from faiss import METRIC_INNER_PRODUCT, METRIC_L2


@dataclass
class FaissIndexerConfig:
    string_factory: str | None = None
    train_size: int | None = None
    metric_type: Literal["inner_product", "l2_distance"] = "inner_product"

    @property
    def metric_type_faiss(self) -> int:
        if self.metric_type == "inner_product":
            return METRIC_INNER_PRODUCT
        elif self.metric_type == "l2_distance":
            return METRIC_L2
        else:
            raise ValueError()
