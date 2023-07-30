import functools
import os
from itertools import chain
from logging import getLogger

from datasets import Dataset

from ettcl.core.config import EvaluatorConfig, EvaluatorMLCConfig
from ettcl.core.evaluate import Evaluator
from ettcl.core.mlc_metrics import MLCMetrics
from ettcl.core.mlknn import MLKNN
from ettcl.core.search import search_dataset
from ettcl.encoding import Encoder
from ettcl.indexing import Indexer
from ettcl.searching import Searcher

logger = getLogger(__name__)


def evaluate_mlc(
    eval_dataset: Dataset,
    index_dataset: Dataset,
    searcher: Searcher,
    index_path: str,
    ks: list[int],
    mlknn_s: float = 1,
    mlknn_k: int = 10,
    metric_key_prefix: str = "eval",
    text_column: str = "text",
    label_column: str = "label",
) -> dict[str, float]:
    if "match_pids" not in index_dataset.column_names:
        index_dataset = search_dataset(
            index_dataset, searcher, index_path, k=mlknn_k, text_column=text_column, report_stats=True
        )
        # index_dataset.save_to_disk(os.path.join(index_path, "index_dataset"))

    index_dataset.set_format("pt")

    mlknn = MLKNN(index_dataset["match_pids"], index_dataset[label_column], k=mlknn_k, s=mlknn_s)
    mlknn.train()
    mlknn.save(os.path.join(index_path, "mlknn"))

    eval_dataset = search_dataset(
        eval_dataset, searcher, index_path, k=mlknn_k, text_column=text_column, report_stats=True
    )
    num_labels_test = max(chain(*map(lambda feat: feat[label_column], eval_dataset))) + 1
    logger.info(f"num_labels_test: {num_labels_test}")
    num_labels = max(mlknn.num_labels, num_labels_test)
    logger.info(f"num_labels: {num_labels}")

    eval_dataset.set_format("pt")
    eval_dataset = eval_dataset.map(lambda pids: {"preds": mlknn.predict(pids)[0]}, input_columns=["match_pids"])

    metric_dict = {}
    for k in ks:
        # postprocess @k
        eval_dataset = eval_dataset.map(lambda preds: {"preds_k": preds[:k]}, input_columns="preds")

        metrics = MLCMetrics(num_labels)
        for batch in eval_dataset.select_columns(["preds_k", label_column]).iter(64):
            metrics.update(list(batch.values()))

        metrics_at_k = metrics.compute()
        metrics_at_k = {f"{metric_key_prefix}_{key}/{k}": val for key, val in metrics_at_k.items()}
        metric_dict |= metrics_at_k

    metrics = MLCMetrics(num_labels)
    for batch in eval_dataset.select_columns(["preds", label_column]).iter(64):
        metrics.update(list(batch.values()))

    metrics = metrics.compute()
    metrics = {f"{metric_key_prefix}_{key}": val for key, val in metrics.items()}
    metric_dict |= metrics

    return metric_dict


class EvaluatorMLC(Evaluator):
    config_cls = EvaluatorMLCConfig
    label_column = "labels"
    evaluate_fn = staticmethod(evaluate_mlc)

    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        config: EvaluatorConfig,
        encoder: Encoder,
        indexer: Indexer,
        searcher: Searcher,
    ) -> None:
        super().__init__(train_dataset, eval_dataset, config, encoder, indexer, searcher)
        self.config: EvaluatorMLCConfig = self.config
        self.evaluate_fn = functools.partial(self.evaluate_fn, mlknn_s=self.config.mlknn_s, mlknn_k=self.config.mlknn_k)
