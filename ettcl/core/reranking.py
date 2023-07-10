import os
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
except ModuleNotFoundError:
    pass
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from ettcl.core.triple_sampling import (
    DataCollatorForTriples,
    ProbabilityType,
    SamplingMethod,
    TripleSamplerDataset,
    TripleSamplingDataBuilder,
)
from ettcl.encoding import Encoder
from ettcl.indexing import Indexer, IndexPath
from ettcl.logging import memory_stats, profile_memory
from ettcl.searching import Searcher
from ettcl.utils.multiprocessing import run_multiprocessed

logger = getLogger(__name__)


@dataclass
class RerankTrainerConfig:
    project: str | None = None
    do_dev_eval: bool = True
    dev_split_size: int | float = 0.1
    do_eval: bool = True
    eval_ks: tuple[int] = (1,)
    resample_interval: int | None = 1
    eval_interval: int | None = None
    dev_eval_interval: int | None = 1
    searcher_sampling_k: int | None = 256
    subsample_train: int | None = None
    subsample_eval: int | None = None
    final_subsample_train: int | None = None
    label_column: str = "label"
    text_column: str = "text"
    remove_columns: list[str] = field(default_factory=list)
    freeze_base_model: bool = False
    sampling_method: SamplingMethod | str = "random"
    probability_type: ProbabilityType | str = "uniform"
    nway: int = 3
    n_positives: int | None = None
    n_negatives: int | None = None
    positive_always_random: bool = False
    lower_ranked_positives: bool = False
    log_model_artifact: bool = False
    stratify_splits: bool = True


class RerankTrainer:
    TEXT_COLUMN = "text"
    LABEL_COLUMN = "label"
    CONFIG_CLS = RerankTrainerConfig
    TRIPLE_SAMPLER_CLS = TripleSamplingDataBuilder

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
            {config.text_column: self.TEXT_COLUMN, config.label_column: self.LABEL_COLUMN}
        )
        self.eval_dataset = eval_dataset
        if self.eval_dataset is not None:
            self.eval_dataset = eval_dataset.remove_columns(self.config.remove_columns)
            self.eval_dataset = self.eval_dataset.rename_columns(
                {config.text_column: self.TEXT_COLUMN, config.label_column: self.LABEL_COLUMN}
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
        while epoch < num_train_epochs:
            logger.info(f"\n\n## EPOCH {epoch}\n")

            with memory_stats():
                pass

            if epoch == next_resample:
                train_subsample = self.subsample(train_dataset, n=self.config.subsample_train)
            if epoch == next_dev_eval or (epoch == next_resample and do_searched_sampling):
                self.index_path = self.build_index(train_subsample, step=global_step)

            if epoch == next_dev_eval:
                next_dev_eval = epoch + dev_eval_interval
                self.evaluate(train_subsample, dev_dataset, epoch, global_step, prefix="eval")

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

            optimizer, scheduler = create_optimizer_and_scheduler(self.model, training_args, len(train_subsample))
            training_args.num_train_epochs = min(next_resample, next_dev_eval)
            trainer = self.trainer_cls(
                model=self.model,
                tokenizer=self.tokenizer,
                args=training_args,
                train_dataset=sampling_dataset,
                data_collator=self.data_collator,
                optimizers=(optimizer, scheduler),
            )

            logger.info(f"training epoch {epoch} - {training_args.num_train_epochs}")
            with memory_stats():
                trainer.train(resume_from_checkpoint=(epoch > 0))  # don't resume in the first epoch
                self.model.cpu()

            # necessary so that reserved memory is freed and can be used by multiprocesses
            torch.cuda.empty_cache()

            global_step = trainer.state.global_step
            epoch = int(round(trainer.state.epoch))

        if self.config.do_dev_eval:
            train_subsample = self.subsample(train_dataset, n=self.config.subsample_train)
            self.index_path = self.build_index(train_subsample, step=global_step)
            self.evaluate(train_subsample, dev_dataset, epoch, global_step, prefix="eval")

        self.latest = Path(training_args.output_dir) / "latest"
        self.latest.mkdir()

        logger.info(f"See training artifacts at {self.latest}")
        trainer.save_model(str(self.latest / "model"))

        train_subsample = self.subsample(self.train_dataset, n=self.config.final_subsample_train)
        self.index_path = self.indexer.index(self.latest / "index", train_subsample[self.TEXT_COLUMN], gpus=True)
        # self.train_dataset.select_columns(self.LABEL_COLUMN).to_pandas().to_feather(self.latest / "train_labels.arrow")

        if self.config.do_eval:
            eval_dataset = self.subsample(self.eval_dataset, self.config.subsample_eval)
            self.evaluate(train_subsample, eval_dataset, prefix="test")

        self.finish()

    def evaluate(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        epoch: int | None = None,
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        logger.info(f"evaluate {prefix}")
        max_k = max(self.config.eval_ks)

        test_dataset = self.search_dataset(test_dataset, self.searcher_eval, k=max_k)

        train_dataset.set_format("pt")
        test_dataset.set_format("pt")

        match_pids = test_dataset["match_pids"]
        if isinstance(match_pids, list):
            logger.warning(f"fewer elements than k={max_k} matched, filling up with (-1).")
            match_pids = torch.nn.utils.rnn.pad_sequence(match_pids, batch_first=True, padding_value=-1)

        match_labels = train_dataset[self.LABEL_COLUMN][match_pids.tolist()]

        logger.info(f"compute metrics {prefix}")

        metrics = {}
        if epoch is not None:
            metrics["train/epoch"] = epoch
        if step is not None:
            metrics["train/step"] = step

        ks = self.config.eval_ks
        for k in ks:
            knn = match_labels[:, :k]
            y_pred = torch.mode(knn)[0]
            assert -1 not in y_pred, "Not enough matches"

            metrics[f"{prefix}/accuracy/{k}"] = accuracy_score(y_pred=y_pred, y_true=test_dataset[self.LABEL_COLUMN])
            metrics[f"{prefix}/precision/micro/{k}"] = precision_score(
                y_pred=y_pred, y_true=test_dataset[self.LABEL_COLUMN], average="micro"
            )
            metrics[f"{prefix}/precision/macro/{k}"] = precision_score(
                y_pred=y_pred, y_true=test_dataset[self.LABEL_COLUMN], average="macro", zero_division=0
            )
            metrics[f"{prefix}/recall/micro/{k}"] = recall_score(
                y_pred=y_pred, y_true=test_dataset[self.LABEL_COLUMN], average="micro"
            )
            metrics[f"{prefix}/recall/macro/{k}"] = recall_score(
                y_pred=y_pred, y_true=test_dataset[self.LABEL_COLUMN], average="macro", zero_division=0
            )
            metrics[f"{prefix}/f1/micro/{k}"] = f1_score(
                y_pred=y_pred, y_true=test_dataset[self.LABEL_COLUMN], average="micro"
            )
            metrics[f"{prefix}/f1/macro/{k}"] = f1_score(
                y_pred=y_pred, y_true=test_dataset[self.LABEL_COLUMN], average="macro", zero_division=0
            )

        metrics[f"{prefix}/accuracy"] = max([metrics[f"{prefix}/accuracy/{k}"] for k in ks])
        metrics[f"{prefix}/precision/micro"] = max([metrics[f"{prefix}/precision/micro/{k}"] for k in ks])
        metrics[f"{prefix}/precision/macro"] = max([metrics[f"{prefix}/precision/macro/{k}"] for k in ks])
        metrics[f"{prefix}/recall/micro"] = max([metrics[f"{prefix}/recall/micro/{k}"] for k in ks])
        metrics[f"{prefix}/recall/macro"] = max([metrics[f"{prefix}/recall/macro/{k}"] for k in ks])
        metrics[f"{prefix}/f1/micro"] = max([metrics[f"{prefix}/f1/micro/{k}"] for k in ks])
        metrics[f"{prefix}/f1/macro"] = max([metrics[f"{prefix}/f1/macro/{k}"] for k in ks])

        logger.info(metrics)
        self.log(metrics)

    @profile_memory
    def build_index(self, dataset: Dataset, step: int) -> IndexPath:
        logger.info("build index")
        index_path = os.path.join(self.training_args.output_dir, f"checkpoint-{step}", "index")
        return self.indexer.index(index_path, dataset[self.TEXT_COLUMN], gpus=True)

    @profile_memory
    def search_dataset(self, dataset: Dataset, searcher: Searcher, k: int) -> Dataset:
        logger.info("search dataset")
        searcher.index_path = self.index_path
        dataset.set_format(None)
        dataset = dataset.map(
            run_multiprocessed(searcher.search),
            input_columns=self.TEXT_COLUMN,
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
        sampling_data_builder = self.TRIPLE_SAMPLER_CLS(
            dataset[self.LABEL_COLUMN],
            sampling_method=self.config.sampling_method,
            probability_type=self.config.probability_type,
            nway=self.config.nway,
            n_positives=self.config.n_positives,
            n_negatives=self.config.n_negatives,
            return_missing=True,
            positive_always_random=self.config.positive_always_random,
            lower_ranked_positives=self.config.lower_ranked_positives,
        )

        dataset = dataset.map(
            sampling_data_builder,
            input_columns=sampling_data_builder.input_columns,
            with_indices=True,
            remove_columns=sampling_data_builder.input_columns,
            desc="Sampling",
            load_from_cache_file=False,
        )

        if sampling_data_builder.return_missing:
            missing_pos = sum(dataset["missing_pos"]) / len(dataset)
            missing_neg = sum(dataset["missing_neg"]) / len(dataset)
            dataset = dataset.remove_columns(["missing_pos", "missing_neg"])
            if missing_pos:
                logger.warning(f"Missing {missing_pos:.3f} positive matches in sampling.")
            if missing_neg:
                logger.warning(f"Missing {missing_neg:.3f} negative matches in sampling.")

        return dataset

    def tokenize(self, dataset: Dataset) -> Dataset:
        logger.info("tokenize")
        return dataset.map(
            lambda batch: self.tokenizer(batch, truncation=True),
            input_columns=self.TEXT_COLUMN,
            remove_columns=self.TEXT_COLUMN,
            batched=True,
            desc="Tokenize",
        )

    def train_dev_split(self) -> tuple[Dataset, Dataset]:
        train_dev_dataset = self.train_dataset.train_test_split(
            self.config.dev_split_size, stratify_by_column=self.LABEL_COLUMN if self.config.stratify_splits else None
        )
        train_dataset = train_dev_dataset["train"]
        dev_dataset = train_dev_dataset["test"]
        return train_dataset, dev_dataset

    def subsample(self, dataset: Dataset, n: int | float | None):
        logger.info("subsample")
        if n is not None:
            return dataset.train_test_split(
                train_size=n,
                stratify_by_column=self.LABEL_COLUMN if self.config.stratify_splits else None,
                load_from_cache_file=False,
            )["train"]
        else:
            return dataset

    def init_wandb(self) -> None:
        logger.info("init wandb")
        output_dir = Path(self.training_args.output_dir)
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

    def log_dir_artifact(self, dir: str, name, type) -> None:
        if hasattr(self, "run"):
            artifact = wandb.Artifact(name, type)
            artifact.add_dir(dir, name=name)
            self.run.log_artifact(artifact)

    def log_file_artifact(self, file: str, name, type) -> None:
        if hasattr(self, "run"):
            artifact = wandb.Artifact(name, type)
            artifact.add_file(file, name=name)
            self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.config.log_model_artifact:
            self.log_dir_artifact(self.latest / "model", name="model", type="model")
            self.log_dir_artifact(self.index_path, name="index", type="index")
            # self.log_file_artifact(self.latest / "train_labels.arrow", name="train_labels", type="labels")

        if hasattr(self, "run"):
            self.run.finish()


def create_optimizer_and_scheduler(model: nn.Module, training_args: TrainingArguments, dataset_size: int):
    """We need to set the number of training steps for the lr scheduler correctly, because RerankTrainer
    is always training only one epoch at a time.
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    num_training_steps = (dataset_size // training_args.train_batch_size) * training_args.num_train_epochs
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer,
        training_args.get_warmup_steps(num_training_steps),
        num_training_steps,
    )

    return optimizer, scheduler
