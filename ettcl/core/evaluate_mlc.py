import os
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import torch

try:
    import wandb
except ModuleNotFoundError:
    pass
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ettcl.core.evaluate import Evaluator, EvaluatorConfig
from ettcl.core.mlc_metrics import MLCMetrics
from ettcl.core.mlknn import MLKNN
from ettcl.encoding import Encoder
from ettcl.indexing import Indexer, IndexPath
from ettcl.searching import Searcher
from ettcl.utils.multiprocessing import run_multiprocessed

logger = getLogger(__name__)


@dataclass
class EvaluatorMLCConfig(EvaluatorConfig):
    label_column: str = "labels"
    stratify_splits: bool = False
    mlknn_s: float = 1

    def __post_init__(self):
        assert len(self.eval_ks) == 1, "Only 1 evaluation k allowed for MLC evaluation"


class EvaluatorMLC(Evaluator):
    LABEL_COLUMN = "labels"
    CONFIG_CLS = EvaluatorMLCConfig

    def _evaluate(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        prefix: str = "",
    ) -> None:
        logger.info(f"evaluate {prefix}")
        k = self.config.eval_ks[0]

        train_dataset = self.search_dataset(train_dataset, self.searcher, k=k)
        test_dataset = self.search_dataset(test_dataset, self.searcher, k=k)

        train_dataset.set_format("pt")
        test_dataset.set_format("pt")

        match_pids = test_dataset["match_pids"]
        if isinstance(match_pids, list):
            logger.warning(f"fewer elements than k={k} matched, filling up with (-1).")
            match_pids = torch.nn.utils.rnn.pad_sequence(match_pids, batch_first=True, padding_value=-1)

        mlknn = MLKNN(train_dataset["match_pids"], train_dataset[self.LABEL_COLUMN], k=k, s=self.config.mlknn_s)
        mlknn.train()

        logger.info(f"compute metrics {prefix}")
        metrics = MLCMetrics(mlknn.num_labels)
        test_dataset = test_dataset.map(lambda pids: {"preds": mlknn.predict(pids)}, input_columns=["match_pids"])

        for batch in test_dataset.select_columns(["preds", self.LABEL_COLUMN]).iter(32):
            metrics.update(list(batch.values()))

        metric_dict = metrics.compute()
        logger.info(metric_dict)
        self.log(metric_dict)
