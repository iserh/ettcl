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

from ettcl.encoding import Encoder
from ettcl.indexing import Indexer, IndexPath
from ettcl.searching import Searcher
from ettcl.utils.multiprocessing import run_multiprocessed

logger = getLogger(__name__)


@dataclass
class EvaluatorConfig:
    output_dir: str | os.PathLike = "evaluations"
    project: str | None = None
    ks: tuple[int] = (1,)
    subsample_train: int | None = None
    subsample_eval: int | None = None
    label_column: str = "label"
    text_column: str = "text"
    prefix: str = "test"


class Evaluator:
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
        self.encoder = encoder
        self.indexer = indexer
        self.searcher = searcher

        self.train_dataset = train_dataset.rename_columns({config.text_column: "text", config.label_column: "label"})
        self.eval_dataset = eval_dataset.rename_columns({config.text_column: "text", config.label_column: "label"})

        self._run_config = {}

    @property
    def run_config(self) -> dict:
        return self._run_config

    @run_config.setter
    def run_config(self, config: dict) -> None:
        self._run_config = config

    def evaluate(self) -> None:
        self.init_wandb()

        train_dataset = self.subsample(self.train_dataset, self.config.subsample_train)
        eval_dataset = self.subsample(self.eval_dataset, self.config.subsample_eval)

        self.index_path = self.build_index(train_dataset)

        self._evaluate(train_dataset, eval_dataset, prefix=self.config.prefix)

        self.finish()

    def _evaluate(self, train_dataset: Dataset, test_dataset: Dataset, prefix: str = "test") -> None:
        logger.info(f"evaluate {prefix}")
        max_k = max(self.config.ks)

        test_dataset = self.search_dataset(test_dataset, self.searcher, k=max_k)

        train_dataset.set_format("pt")
        test_dataset.set_format("pt")

        match_pids = test_dataset["match_pids"]
        if isinstance(match_pids, list):
            logger.warning(f"fewer elements than k={max_k} matched, filling up with (-1).")
            match_pids = torch.nn.utils.rnn.pad_sequence(match_pids, batch_first=True, padding_value=-1)

        match_labels = train_dataset["label"][match_pids.tolist()]

        logger.info(f"compute metrics {prefix}")
        metrics = {}
        for k in self.config.ks:
            knn = match_labels[:, :k]
            y_pred = torch.mode(knn)[0]
            assert -1 not in y_pred, "Not enough matches"

            metrics[f"{prefix}/accuracy/{k}"] = accuracy_score(y_pred=y_pred, y_true=test_dataset["label"])
            metrics[f"{prefix}/precision/micro/{k}"] = precision_score(
                y_pred=y_pred, y_true=test_dataset["label"], average="micro"
            )
            metrics[f"{prefix}/precision/macro/{k}"] = precision_score(
                y_pred=y_pred, y_true=test_dataset["label"], average="macro"
            )
            metrics[f"{prefix}/recall/micro/{k}"] = recall_score(
                y_pred=y_pred, y_true=test_dataset["label"], average="micro"
            )
            metrics[f"{prefix}/recall/macro/{k}"] = recall_score(
                y_pred=y_pred, y_true=test_dataset["label"], average="macro"
            )
            metrics[f"{prefix}/f1/micro/{k}"] = f1_score(y_pred=y_pred, y_true=test_dataset["label"], average="micro")
            metrics[f"{prefix}/f1/macro/{k}"] = f1_score(y_pred=y_pred, y_true=test_dataset["label"], average="macro")

        logger.info(metrics)
        self.log(metrics)

    def build_index(self, dataset: Dataset) -> IndexPath:
        logger.info("build index")
        index_path = os.path.join(self.config.output_dir, "index_eval")
        return self.indexer.index(index_path, dataset["text"], gpus=True)

    def search_dataset(self, dataset: Dataset, searcher: Searcher, k: int) -> Dataset:
        logger.info("search dataset")
        searcher.index_path = self.index_path
        dataset.set_format(None)
        dataset = dataset.map(
            run_multiprocessed(searcher.search),
            input_columns="text",
            fn_kwargs={"k": k},
            batched=True,
            num_proc=torch.cuda.device_count(),
            with_rank=True,
            desc="Searching",
        )

        # log some statistics about how many matches were found
        dataset.set_format("numpy")
        dataset = dataset.map(lambda pids: {"len_": len(pids)}, input_columns="match_pids")
        avg_matches = dataset["len_"].mean()
        dataset = dataset.remove_columns("len_")
        logger.info(f"average #matches: {avg_matches}")

        return dataset

    def subsample(self, dataset: Dataset, n: int | float | None):
        logger.info("subsample")
        if n is not None:
            return dataset.train_test_split(train_size=n, stratify_by_column="label", load_from_cache_file=False)[
                "train"
            ]
        else:
            return dataset

    def init_wandb(self) -> None:
        logger.info("init wandb")
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True)
        try:
            self.run = wandb.init(
                project=self.config.project,
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
