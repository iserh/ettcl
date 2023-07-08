from logging import getLogger

import torch
from datasets import Dataset
from dataclasses import dataclass

from ettcl.core.reranking import RerankTrainer, RerankTrainerConfig
from ettcl.core.triple_sampling import TripleSamplingDataBuilderMLC
from ettcl.core.mlc_metrics import MLCMetrics
from ettcl.core.mlknn import MLKNN

logger = getLogger(__name__)

@dataclass
class RerankMLCTrainerConfig(RerankTrainerConfig):
    label_column: str = "labels"
    stratify_splits: bool = False
    mlknn_s: float = 1

    def __post_init__(self):
        assert len(self.eval_ks) == 1, "Only 1 evaluation k allowed for MLC evaluation"


class RerankMLCTrainer(RerankTrainer):
    LABEL_COLUMN = "labels"
    CONFIG_CLS = RerankMLCTrainerConfig
    TRIPLE_SAMPLER_CLS = TripleSamplingDataBuilderMLC

    def evaluate(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        epoch: int | None = None,
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        logger.info(f"evaluate {prefix}")
        k = self.config.eval_ks[0]

        train_dataset = self.search_dataset(train_dataset, self.searcher_eval, k=k)
        test_dataset = self.search_dataset(test_dataset, self.searcher_eval, k=k)

        train_dataset.set_format("pt")
        test_dataset.set_format("pt")

        self.mlknn = MLKNN(train_dataset["match_pids"], train_dataset[self.LABEL_COLUMN], k=k, s=self.config.mlknn_s)
        self.mlknn.train()

        logger.info(f"compute metrics {prefix}")
        metrics = MLCMetrics(self.mlknn.num_labels)
        test_dataset = test_dataset.map(lambda pids: {"preds": self.mlknn.predict(pids)}, input_columns=["match_pids"])

        for batch in test_dataset.select_columns(["preds", self.LABEL_COLUMN]).iter(32):
            metrics.update(list(batch.values()))

        metric_dict = metrics.compute()

        if epoch is not None:
            metric_dict["train/epoch"] = epoch
        if step is not None:
            metric_dict["train/step"] = step

        logger.info(metric_dict)
        self.log(metric_dict)
