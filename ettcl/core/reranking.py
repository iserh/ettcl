import os
from dataclasses import dataclass, field
from logging import getLogger

import torch
try:
    import wandb
except ModuleNotFoundError:
    pass
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import PreTrainedTokenizerBase, Trainer, TrainingArguments, PreTrainedModel

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
    resample_interval: int = 1
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
    fill_missing: int | None = None


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
        self.train_dataset = self.train_dataset.rename_columns({config.text_column: "text", config.label_column: "label"})
        self.eval_dataset = eval_dataset
        if self.eval_dataset is not None:
            self.eval_dataset = eval_dataset.remove_columns(self.config.remove_columns)
            self.eval_dataset = self.eval_dataset.rename_columns({config.text_column: "text", config.label_column: "label"})
        else:
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

        if self.config.do_dev_eval:
            train_dataset, dev_dataset = self.train_dev_split()
            logger.info(f"dev_dataset size: {len(dev_dataset)}")

        logger.info(f"train_dataset size: {len(train_dataset)/(self.config.subsample_train or 1)}")
        if self.config.do_eval:
            logger.info(f"eval_dataset size: {len(self.eval_dataset)/(self.config.subsample_eval or 1)}")

        self.init_wandb()

        epoch, global_step = 0, 0
        while epoch < num_train_epochs:
            logger.info(f"\n\n## EPOCH {epoch}\n")

            train_subsample = self.subsample(train_dataset, n=self.config.subsample_train)
            if do_searched_sampling or self.config.do_dev_eval:
                self.index_path = self.build_index(train_subsample, step=global_step)
            if self.config.do_dev_eval:
                self.evaluate(train_subsample, dev_dataset, prefix="dev")
            if do_searched_sampling:
                train_subsample = self.search_dataset(
                    train_subsample, self.searcher_sampling, self.config.searcher_sampling_k
                )

            train_subsample = self.build_sampling_data(train_subsample)
            train_subsample = self.tokenize(train_subsample)
            train_subsample = TripleSamplerDataset(train_subsample, self.config.nway)

            if self.config.freeze_base_model:
                self.model.freeze_base_model()

            training_args.num_train_epochs = min(epoch + self.config.resample_interval, num_train_epochs)
            trainer = self.trainer_cls(
                model=self.model,
                tokenizer=self.tokenizer,
                args=training_args,
                train_dataset=train_subsample,
                data_collator=self.data_collator,
            )

            logger.info(f"training epoch {epoch} - {training_args.num_train_epochs}")
            trainer.train(resume_from_checkpoint=(epoch > 0))  # don't resume in the first epoch

            global_step = trainer.state.global_step
            epoch = trainer.state.epoch

            train_subsample = self.subsample(train_dataset, n=self.config.subsample_train)

            if do_searched_sampling or self.config.do_dev_eval:
                index = self.build_index(train_subsample, step=global_step)
            if do_searched_sampling:
                self.searcher_sampling.index = index

        if self.config.do_eval or self.config.do_dev_eval:
            self.index_path = self.build_index(train_dataset, step=global_step)
        if self.config.do_dev_eval:
            self.evaluate(train_dataset, dev_dataset, prefix="dev")
        if self.config.do_eval:
            eval_dataset = self.subsample(self.eval_dataset, self.config.subsample_eval)
            self.evaluate(train_dataset, eval_dataset, prefix="test")

    def tokenize(self, dataset: Dataset) -> Dataset:
        logger.info("tokenize")
        return dataset.map(
            lambda batch: self.tokenizer(batch, truncation=True),
            input_columns=self.config.text_column,
            remove_columns=self.config.text_column,
            batched=True,
            desc="Tokenize",
        )

    def build_sampling_data(
        self,
        dataset: Dataset,
    ):
        logger.info("build sampling data")
        sampling_data_builder = TripleSamplingDataBuilder(
            dataset["label"],
            sampling_method=self.config.sampling_method,
            probability_type=self.config.probability_type,
            nway=self.config.nway,
            fill_missing=self.config.fill_missing,
            return_missing=True,
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

    def search_dataset(self, dataset: Dataset, searcher: Searcher, k: int) -> Dataset:
        logger.info("search dataset")
        searcher.index_path = self.index_path
        dataset.set_format(None)
        return dataset.map(
            run_multiprocessed(searcher.search),
            input_columns=self.config.text_column,
            fn_kwargs={"k": k},
            batched=True,
            num_proc=torch.cuda.device_count(),
            with_rank=True,
            desc="Searching",
        )

    def train_dev_split(self) -> tuple[Dataset, Dataset]:
        train_dev_dataset = self.train_dataset.train_test_split(
            self.config.dev_split_size, stratify_by_column="label"
        )
        train_dataset = train_dev_dataset["train"]
        dev_dataset = train_dev_dataset["test"]
        return train_dataset, dev_dataset

    def subsample(self, dataset: Dataset, n: int | float | None):
        logger.info("subsample")
        if n is not None:
            return dataset.train_test_split(train_size=n, stratify_by_column="label")["train"]
        else:
            return dataset

    def build_index(self, dataset: Dataset, step: int) -> IndexPath:
        logger.info("build index")
        index_path = os.path.join(self.training_args.output_dir, f"checkpoint-{step}", "index")
        return self.indexer.index(index_path, dataset[self.config.text_column], gpus=True)

    def init_wandb(self) -> None:
        logger.info("init wandb")
        try:
            self.run = wandb.init(
                dir=self.training_args.output_dir,
                config=self.run_config,
            )
        except ModuleNotFoundError:
            pass

    def log(self, values: dict) -> None:
        if hasattr(self, "run"):
            self.run.log(values)

    def evaluate(self, train_dataset: Dataset, test_dataset: Dataset, prefix: str = "") -> None:
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
        metrics = {}
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
