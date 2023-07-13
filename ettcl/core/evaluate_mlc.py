import functools
import os
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
from itertools import chain

logger = getLogger(__name__)


def evaluate_mlc(
    eval_dataset: Dataset,
    index_dataset: Dataset,
    searcher: Searcher,
    index_path: str,
    ks: list[int],
    mlknn_s: float = 1,
    metric_key_prefix: str = "eval",
    text_column: str = "text",
    label_column: str = "label",
) -> dict[str, float]:
    assert len(ks) == 1, "MLC evaluation only possible with single k"
    k = ks[0]

    if "match_pids" not in index_dataset.column_names:
        index_dataset = search_dataset(
            index_dataset, searcher, index_path, k=k, text_column=text_column, report_stats=True
        )

    index_dataset.set_format("pt")

    mlknn = MLKNN(index_dataset["match_pids"], index_dataset[label_column], k=k, s=mlknn_s)
    mlknn.train()
    mlknn.save(os.path.join(index_path, "mlknn"))

    eval_dataset = search_dataset(eval_dataset, searcher, index_path, k=k, text_column=text_column, report_stats=True)
    num_labels_test = max(chain(*map(lambda feat: feat[label_column], eval_dataset))) + 1
    logger.info(f"num_labels_test: {num_labels_test}")
    num_labels = max(mlknn.num_examples, num_labels_test)
    logger.info(f"num_labels: {num_labels}")

    eval_dataset.set_format("pt")
    metrics = MLCMetrics(num_labels)
    eval_dataset = eval_dataset.map(lambda pids: {"preds": mlknn.predict(pids)}, input_columns=["match_pids"])

    for batch in eval_dataset.select_columns(["preds", label_column]).iter(32):
        metrics.update(list(batch.values()))

    metric_dict = metrics.compute()
    metric_dict = {f"{metric_key_prefix}_{k}": v for k, v in metric_dict.items()}

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
        self.evaluate_fn = functools.partial(self.evaluate_fn, mlknn_s=self.config.mlknn_s)
