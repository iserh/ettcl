import os
from logging import getLogger
from pathlib import Path

import torch

try:
    import wandb
except ModuleNotFoundError:
    pass
from datasets import Dataset, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ettcl.core.config import EvaluatorConfig
from ettcl.core.search import search_dataset
from ettcl.core.triple_sampling import subsample
from ettcl.encoding import Encoder
from ettcl.indexing import Indexer
from ettcl.searching import Searcher

logger = getLogger(__name__)


def evaluate(
    eval_dataset: Dataset,
    index_dataset: Dataset,
    searcher: Searcher,
    index_path: str,
    ks: list[int],
    metric_key_prefix: str = "eval",
    text_column: str = "text",
    label_column: str = "label",
) -> dict[str, float]:
    max_k = max(ks)

    eval_dataset = search_dataset(
        eval_dataset, searcher, index_path, k=max_k, text_column=text_column, report_stats=True
    )

    eval_dataset.set_format("torch")
    index_dataset.set_format("torch")

    index_labels = index_dataset[label_column]

    match_pids = eval_dataset["match_pids"]
    if isinstance(match_pids, list):
        logger.warning(f"fewer elements than k={max_k} matched, filling up with (-1).")
        match_pids = torch.nn.utils.rnn.pad_sequence(match_pids, batch_first=True, padding_value=-1)

    match_labels = index_labels[match_pids.tolist()]
    match_labels[match_pids == -1] = -1

    logger.info(f"compute metrics {metric_key_prefix}")

    metrics = {}

    for k in ks:
        knn = match_labels[:, :k]
        y_pred = torch.mode(knn)[0]
        if -1 in y_pred:
            logger.warning("Not enough matches")

        metrics[f"{metric_key_prefix}_accuracy/{k}"] = accuracy_score(y_pred=y_pred, y_true=eval_dataset[label_column])
        metrics[f"{metric_key_prefix}_precision/micro/{k}"] = precision_score(
            y_pred=y_pred, y_true=eval_dataset[label_column], average="micro"
        )
        metrics[f"{metric_key_prefix}_precision/macro/{k}"] = precision_score(
            y_pred=y_pred, y_true=eval_dataset[label_column], average="macro", zero_division=0
        )
        metrics[f"{metric_key_prefix}_recall/micro/{k}"] = recall_score(
            y_pred=y_pred, y_true=eval_dataset[label_column], average="micro"
        )
        metrics[f"{metric_key_prefix}_recall/macro/{k}"] = recall_score(
            y_pred=y_pred, y_true=eval_dataset[label_column], average="macro", zero_division=0
        )
        metrics[f"{metric_key_prefix}_f1/micro/{k}"] = f1_score(
            y_pred=y_pred, y_true=eval_dataset[label_column], average="micro"
        )
        metrics[f"{metric_key_prefix}_f1/macro/{k}"] = f1_score(
            y_pred=y_pred, y_true=eval_dataset[label_column], average="macro", zero_division=0
        )

    metrics[f"{metric_key_prefix}_accuracy"] = max([metrics[f"{metric_key_prefix}_accuracy/{k}"] for k in ks])
    metrics[f"{metric_key_prefix}_precision/micro"] = max(
        [metrics[f"{metric_key_prefix}_precision/micro/{k}"] for k in ks]
    )
    metrics[f"{metric_key_prefix}_precision/macro"] = max(
        [metrics[f"{metric_key_prefix}_precision/macro/{k}"] for k in ks]
    )
    metrics[f"{metric_key_prefix}_recall/micro"] = max([metrics[f"{metric_key_prefix}_recall/micro/{k}"] for k in ks])
    metrics[f"{metric_key_prefix}_recall/macro"] = max([metrics[f"{metric_key_prefix}_recall/macro/{k}"] for k in ks])
    metrics[f"{metric_key_prefix}_f1/micro"] = max([metrics[f"{metric_key_prefix}_f1/micro/{k}"] for k in ks])
    metrics[f"{metric_key_prefix}_f1/macro"] = max([metrics[f"{metric_key_prefix}_f1/macro/{k}"] for k in ks])

    return metrics


class Evaluator:
    config_cls = EvaluatorConfig
    evaluate_fn = staticmethod(evaluate)

    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        config: EvaluatorConfig,
        encoder: Encoder,
        indexer: Indexer,
        searcher: Searcher,
    ) -> None:
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.text_column = config.text_column
        self.label_column = config.label_column
        self.encoder = encoder
        self.indexer = indexer
        self.searcher = searcher

        self.train_dataset = train_dataset.rename_columns(
            {config.text_column: self.text_column, config.label_column: self.label_column}
        )
        self.eval_dataset = eval_dataset.rename_columns(
            {config.text_column: self.text_column, config.label_column: self.label_column}
        )

        self._run_config = {}

    @property
    def run_config(self) -> dict:
        return self._run_config

    @run_config.setter
    def run_config(self, config: dict) -> None:
        self._run_config = config

    def evaluate(self) -> None:
        self.init_wandb()

        index_dataset = subsample(
            self.train_dataset, self.config.subsample_train, self.config.stratify_splits, self.label_column
        )
        eval_dataset = subsample(
            self.eval_dataset, self.config.subsample_eval, self.config.stratify_splits, self.label_column
        )

        index_path = os.path.join(self.config.output_dir, "index_eval")
        if not os.path.exists(index_path) and self.config.reuse_index:
            logger.info("build index")
            self.indexer.index(index_path, index_dataset[self.text_column], gpus=True)

        index_dataset_path = os.path.join(index_path, "index_dataset")
        if os.path.exists(index_dataset_path) and self.config.reuse_index:
            logger.info(f"load index dataset from {index_dataset_path}")
            index_dataset = load_from_disk(index_dataset_path)

        logger.info(f"compute metrics")
        metrics = self.evaluate_fn(
            eval_dataset=eval_dataset,
            index_dataset=index_dataset,
            searcher=self.searcher,
            index_path=index_path,
            ks=self.config.eval_ks,
            metric_key_prefix=self.config.prefix,
            text_column=self.text_column,
            label_column=self.label_column,
        )

        metrics = {k.replace(f"{self.config.prefix}_", f"{self.config.prefix}/"): v for k, v in metrics.items()}
        metrics |= {"train/global_step": 0, "train/epoch": 0}

        logger.info(metrics)
        self.log(metrics)

        self.finish()

    def init_wandb(self) -> None:
        logger.info("init wandb")
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exp_name = self.config.exp_name or f"{self.run_config['dataset']}-{self.run_config['architecture']}"
        try:
            if self.config.resume is not None:
                self.run = wandb.init(
                    id=self.config.resume,
                    project=self.config.project,
                    dir=output_dir,
                    resume=True,
                )
            else:
                self.run = wandb.init(
                    project=self.config.project,
                    name=exp_name,
                    dir=output_dir,
                    config=self.run_config,
                    save_code=True,
                )
            self.run.log_code(
                ".",
                include_fn=lambda path: path.endswith(".py")
                or path.endswith(".cpp")
                or path.endswith(".cu")
                or path.endswith(".yml"),
            )
        except ModuleNotFoundError:
            pass

    def log(self, values: dict) -> None:
        if hasattr(self, "run"):
            self.run.log(values)

    def finish(self) -> None:
        if hasattr(self, "run"):
            self.run.finish()
