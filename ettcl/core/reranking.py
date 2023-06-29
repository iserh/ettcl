import os
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

import numpy as np
import torch

try:
    import wandb
except ModuleNotFoundError:
    pass
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments

from ettcl.core.triple_sampling import (
    DataCollatorForTriples,
    ProbabilityType,
    SamplingMethod,
    TripleSamplerDataset,
    TripleSamplingDataBuilder,
)
from ettcl.encoding import Encoder
from ettcl.indexing import Indexer, IndexPath
from ettcl.searching import Searcher
from ettcl.utils.multiprocessing import run_multiprocessed

logger = getLogger(__name__)


@dataclass
class RerankTrainerConfig:
    do_dev_eval: bool = False
    dev_split_size: int | float = 0.2
    do_eval: bool = False
    eval_ks: tuple[int] = (1,)
    resample_interval: int | None = None
    eval_interval: int | None = None
    dev_eval_interval: int | None = None
    searcher_sampling_k: int | None = None
    subsample_train: int | None = None
    subsample_eval: int | None = None
    label_column: str = "label"
    text_column: str = "text"
    remove_columns: list[str] = field(default_factory=list)
    freeze_base_model: bool = False
    sampling_method: SamplingMethod | str = "random"
    probability_type: ProbabilityType | str = "uniform"
    nway: int = 2
    n_positives: int | None = None
    n_negatives: int | None = None
    positive_always_random: bool = False
    log_model_artifact: bool = True


class RerankTrainer:
    def __init__(
        self,
        trainer_cls: type[Trainer],
        model: PreTrainedModel,
        encoder: Encoder,
        tokenizer: PreTrainedTokenizerBase,
        config: RerankTrainerConfig,
        training_args: TrainingArguments,
        train_dataset: Dataset,
        indexer: Indexer,
        searcher_eval: Searcher | None = None,
        eval_dataset: Dataset | None = None,
        searcher_sampling: Searcher | None = None,
    ) -> None:
        self.model = model
        self.trainer_cls = trainer_cls
        self.tokenizer = tokenizer
        self.config = config
        self.training_args = training_args
        self.train_dataset = train_dataset.remove_columns(self.config.remove_columns)
        self.train_dataset = self.train_dataset.rename_columns(
            {config.text_column: "text", config.label_column: "label"}
        )
        self.eval_dataset = eval_dataset
        if self.eval_dataset is not None:
            self.eval_dataset = eval_dataset.remove_columns(self.config.remove_columns)
            self.eval_dataset = self.eval_dataset.rename_columns(
                {config.text_column: "text", config.label_column: "label"}
            )
        elif self.config.do_eval:
            logger.warning("No evaluation dataset provided, disabling final evaluation.")
            self.config.do_eval = False
        self.encoder = encoder
        self.indexer = indexer
        self.searcher_eval = searcher_eval
        if self.searcher_eval is None:
            self.config.do_eval = False
            self.config.do_dev_eval = False
        self.searcher_sampling = searcher_sampling
        if self.config.sampling_method == "searched" and self.searcher_sampling is None:
            raise RuntimeError("If sampling_mode == `searched` you need to provide the argument `searcher_eval`")

        self.data_collator = DataCollatorForTriples(self.tokenizer)
        self._run_config = {}

    @property
    def run_config(self) -> dict:
        return self._run_config

    @run_config.setter
    def run_config(self, config: dict) -> None:
        self._run_config = config

    def train(self) -> None:
        training_args = self.training_args
        num_train_epochs = training_args.num_train_epochs
        do_searched_sampling = self.config.sampling_method == "searched"
        resample_interval = self.config.resample_interval or num_train_epochs
        dev_eval_interval = self.config.dev_eval_interval if self.config.do_dev_eval else None
        eval_interval = self.config.eval_interval if self.config.do_eval else None

        train_dataset = self.train_dataset

        if self.config.do_dev_eval:
            train_dataset, dev_dataset = self.train_dev_split()
            logger.info(f"dev_dataset size: {len(dev_dataset)}")

        logger.info(f"train_dataset size: {len(train_dataset)*(self.config.subsample_train or 1)}")
        if self.config.do_eval:
            logger.info(f"eval_dataset size: {len(self.eval_dataset)*(self.config.subsample_eval or 1)}")

        self.init_wandb()

        epoch, global_step = 0, 0
        next_resample = 0
        next_dev_eval = 0 if dev_eval_interval is not None else num_train_epochs
        next_eval = 0 if eval_interval is not None else num_train_epochs
        while epoch < num_train_epochs:
            logger.info(f"\n\n## EPOCH {epoch}\n")

            if epoch == next_resample:
                train_subsample = self.subsample(train_dataset, n=self.config.subsample_train)
            if epoch in [next_dev_eval, next_eval] or (epoch == next_resample and do_searched_sampling):
                self.index_path = self.build_index(train_subsample, step=global_step)

            if epoch == next_dev_eval:
                next_dev_eval = epoch + dev_eval_interval
                self.evaluate(train_subsample, dev_dataset, epoch, global_step, prefix="dev")
            if epoch == next_eval:
                next_eval = epoch + eval_interval
                eval_dataset = self.subsample(self.eval_dataset, self.config.subsample_eval)
                self.evaluate(train_dataset, eval_dataset, epoch, global_step, prefix="test")

            if epoch == next_resample:
                next_resample = epoch + resample_interval
                if do_searched_sampling:
                    train_subsample = self.search_dataset(
                        train_subsample, self.searcher_sampling, self.config.searcher_sampling_k
                    )
                sampling_dataset = self.build_sampling_data(train_subsample)
                sampling_dataset = self.tokenize(sampling_dataset)
                sampling_dataset = TripleSamplerDataset(sampling_dataset, self.config.nway)

            if self.config.freeze_base_model:
                logger.info("Freezing base model parameters.")
                self.model.freeze_base_model()

            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logger.info(f"Training {n_params} parameters")

            training_args.num_train_epochs = min(next_resample, next_dev_eval, next_eval)
            trainer = self.trainer_cls(
                model=self.model,
                tokenizer=self.tokenizer,
                args=training_args,
                train_dataset=sampling_dataset,
                data_collator=self.data_collator,
            )

            logger.info(f"training epoch {epoch} - {training_args.num_train_epochs}")
            trainer.train(resume_from_checkpoint=(epoch > 0))  # don't resume in the first epoch

            global_step = trainer.state.global_step
            epoch = int(trainer.state.epoch)

        if self.config.do_eval or self.config.do_dev_eval:
            self.index_path = self.build_index(train_dataset, step=global_step)

        if self.config.log_model_artifact:
            latest = os.path.join(training_args.output_dir, "latest")
            trainer.save_model(latest)
            self.log_dir_artifact(latest, name="model", type="model")
            if self.index_path is not None:
                self.log_dir_artifact(self.index_path, name="index", type="index")

        if self.config.do_dev_eval:
            self.evaluate(train_dataset, dev_dataset, epoch, global_step, prefix="dev")
        if self.config.do_eval:
            eval_dataset = self.subsample(self.eval_dataset, self.config.subsample_eval)
            self.evaluate(train_dataset, eval_dataset, epoch, global_step, prefix="test")

        self.model.cpu()

    def evaluate(self, train_dataset: Dataset, test_dataset: Dataset, epoch: int, step: int, prefix: str = "") -> None:
        logger.info(f"evaluate {prefix}")
        max_k = max(self.config.eval_ks)

        test_dataset = self.search_dataset(test_dataset, self.searcher_eval, k=max_k)

        train_dataset.set_format("pt")
        test_dataset.set_format("pt")

        match_pids = test_dataset["match_pids"]
        if isinstance(match_pids, list):
            logger.warning(f"fewer elements than k={max_k} matched, filling up with (-1).")
            match_pids = torch.nn.utils.rnn.pad_sequence(match_pids, batch_first=True, padding_value=-1)

        match_labels = train_dataset["label"][match_pids.tolist()]

        logger.info(f"compute metrics {prefix}")
        metrics = {"train/epoch": epoch, "train/step": step}
        for k in self.config.eval_ks:
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

    def build_index(self, dataset: Dataset, step: int) -> IndexPath:
        logger.info("build index")
        index_path = os.path.join(self.training_args.output_dir, f"checkpoint-{step}", "index")
        return self.indexer.index(index_path, dataset[self.config.text_column], gpus=True)

    def search_dataset(self, dataset: Dataset, searcher: Searcher, k: int) -> Dataset:
        logger.info("search dataset")
        searcher.index_path = self.index_path
        dataset.set_format(None)
        dataset = dataset.map(
            run_multiprocessed(searcher.search),
            input_columns=self.config.text_column,
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

    def build_sampling_data(
        self,
        dataset: Dataset,
    ):
        logger.info("build sampling data")
        dataset.set_format(None)
        sampling_data_builder = TripleSamplingDataBuilder(
            dataset["label"],
            sampling_method=self.config.sampling_method,
            probability_type=self.config.probability_type,
            nway=self.config.nway,
            n_positives=self.config.n_positives,
            n_negatives=self.config.n_negatives,
            return_missing=True,
            positive_always_random=self.config.positive_always_random,
        )

        dataset = dataset.map(
            sampling_data_builder,
            input_columns=sampling_data_builder.input_columns,
            with_indices=True,
            remove_columns=sampling_data_builder.input_columns,
            desc="Sampling",
        )

        if sampling_data_builder.return_missing:
            missing_pos = sum(dataset["missing_pos"])
            missing_neg = sum(dataset["missing_neg"])
            dataset = dataset.remove_columns(["missing_pos", "missing_neg"])
            if missing_pos:
                logger.warning(f"Missing {missing_pos} positive matches in sampling.")
            if missing_neg:
                logger.warning(f"Missing {missing_neg} negative matches in sampling.")

        return dataset

    def tokenize(self, dataset: Dataset) -> Dataset:
        logger.info("tokenize")
        return dataset.map(
            lambda batch: self.tokenizer(batch, truncation=True),
            input_columns=self.config.text_column,
            remove_columns=self.config.text_column,
            batched=True,
            desc="Tokenize",
        )

    def train_dev_split(self) -> tuple[Dataset, Dataset]:
        train_dev_dataset = self.train_dataset.train_test_split(self.config.dev_split_size, stratify_by_column="label")
        train_dataset = train_dev_dataset["train"]
        dev_dataset = train_dev_dataset["test"]
        return train_dataset, dev_dataset

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
        output_dir = Path(self.training_args.output_dir)
        output_dir.mkdir(parents=True)
        try:
            self.run = wandb.init(
                dir=output_dir,
                config=self.run_config,
                save_code=True,
            )
        except ModuleNotFoundError:
            pass

    def log(self, values: dict) -> None:
        if hasattr(self, "run"):
            self.run.log(values)

    def log_dir_artifact(self, dir: str, name, type) -> None:
        if hasattr(self, "run"):
            artifact = wandb.Artifact(name, type)
            artifact.add_dir(dir, name=name)
            self.run.log_artifact(artifact)

    def finish(self) -> None:
        if hasattr(self, "run"):
            self.run.finish()
