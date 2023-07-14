import os
import shutil
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

try:
    import wandb
except ModuleNotFoundError:
    pass
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from ettcl.core.config import RerankTrainerConfig
from ettcl.core.evaluate import evaluate
from ettcl.core.triple_sampling import (
    DataCollatorForTriples,
    TripleSamplerDataset,
    TripleSamplingDataBuilder,
    subsample,
)
from ettcl.encoding import Encoder
from ettcl.indexing import Indexer
from ettcl.logging import memory_stats
from ettcl.searching import Searcher

logger = getLogger(__name__)


class ResampleCallback(TrainerCallback):
    def __init__(
        self,
        train_dataset: Dataset,
        indexer: Indexer,
        searcher: Searcher,
        config: RerankTrainerConfig,
    ) -> None:
        self.train_dataset = train_dataset
        self.indexer = indexer
        self.searcher = searcher
        self.config = config
        self.next_resample = self.config.resample_interval
        self.next_dev_eval = self.config.dev_eval_interval

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataloader: DataLoader,
        **kwargs,
    ):
        if self.next_resample is None:
            return
        elif self.next_dev_eval is None:
            return

        cur_epoch = int(state.epoch)

        should_rebuild_index = cur_epoch in [self.next_resample, self.next_dev_eval]

        if cur_epoch == self.next_dev_eval:
            self.next_dev_eval = cur_epoch + self.config.dev_eval_interval
        if cur_epoch == self.next_resample:
            self.next_resample = cur_epoch + self.config.resample_interval
            should_subsample, should_resample = True, True
        else:
            should_subsample, should_resample = False, False

        device = model.device
        model.cpu()
        # necessary so that reserved memory is freed and can be used by multiprocesses
        torch.cuda.empty_cache()

        train_dataset = self.train_dataset

        if should_subsample:
            logger.info("subsampling train dataset")
            train_dataset = subsample(
                self.train_dataset,
                self.config.subsample_train,
                self.config.stratify_splits,
                self.config.label_column,
            )

        if should_rebuild_index:
            logger.info("build index")
            index_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", "index")
            self.indexer.index(index_path, train_dataset[self.config.text_column], gpus=True)
            train_dataset.save_to_disk(os.path.join(index_path, "index_dataset"))

        if should_resample:
            train_dataloader.dataset.create_sampling_data(
                dataset=train_dataset,
                searcher=self.searcher,
                tokenizer=tokenizer,
                index_path=index_path,
            )

        model.to(device)


class TrainerWithEvaluation(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | nn.Module = None,
        args: TrainingArguments = None,
        data_collator: Any | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], Dict] | None = None,
        callbacks: List[TrainerCallback] | None = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        evaluate_fn: Callable | None = None,
        searcher: Searcher | None = None,
        eval_ks: tuple[int] = (1,),
        text_column: str = "text",
        label_column: str = "label",
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        assert searcher is not None, "Must provide searcher."

        self.evaluate_fn = evaluate_fn
        self.searcher = searcher
        self.eval_ks = eval_ks
        self.text_column = text_column
        self.label_column = label_column

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: List[str] | None = None,
        metric_key_prefix: str = "eval",
        index_path: str | None = None,
    ) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if index_path is None:
            index_path = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}", "index")

        index_dataset_path = os.path.join(index_path, "index_dataset")
        index_dataset = load_from_disk(index_dataset_path)

        logger.info(f"evaluate {metric_key_prefix}")
        self._memory_tracker.start()

        device = self.model.device
        self.model.cpu()
        torch.cuda.empty_cache()

        metrics = self.evaluate_fn(
            eval_dataset=eval_dataset,
            index_dataset=index_dataset,
            searcher=self.searcher,
            index_path=index_path,
            ks=self.eval_ks,
            metric_key_prefix=metric_key_prefix,
            text_column=self.text_column,
            label_column=self.label_column,
        )

        self.model.to(device)

        self.log(metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics


class RerankTrainer:
    config_cls = RerankTrainerConfig
    text_column = "text"
    label_column = "label"
    triples_sampler_cls = TripleSamplingDataBuilder
    evaluate_fn = staticmethod(evaluate)

    def __init__(
        self,
        model: PreTrainedModel,
        encoder: Encoder,
        tokenizer: PreTrainedTokenizerBase,
        config: RerankTrainerConfig,
        training_args: TrainingArguments,
        train_dataset: Dataset,
        indexer: Indexer,
        searcher_eval: Searcher | None = None,
        val_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        searcher_sampling: Searcher | None = None,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.config = config
        self.training_args = training_args
        self.indexer = indexer
        self.searcher_eval = searcher_eval
        if self.searcher_eval is None:
            self.config.do_eval = False
            self.config.do_dev_eval = False
            self.config.dev_eval_interval = None

        self.searcher_sampling = searcher_sampling
        if self.config.sampling_method == "searched" and self.searcher_sampling is None:
            raise RuntimeError("If sampling_mode == `searched` you need to provide the argument `searcher_eval`")

        self.train_dataset = train_dataset.remove_columns(self.config.remove_columns)
        self.train_dataset = self.train_dataset.rename_columns(
            {config.text_column: self.text_column, config.label_column: self.label_column}
        )

        self.val_dataset = val_dataset
        if self.val_dataset is not None:
            self.val_dataset = self.val_dataset.remove_columns(self.config.remove_columns)
            self.val_dataset = self.val_dataset.rename_columns(
                {config.text_column: self.text_column, config.label_column: self.label_column}
            )
        else:
            self.config.do_dev_eval = False
            self.config.dev_eval_interval = None

        self.eval_dataset = eval_dataset
        if self.eval_dataset is not None:
            self.eval_dataset = eval_dataset.remove_columns(self.config.remove_columns)
            self.eval_dataset = self.eval_dataset.rename_columns(
                {config.text_column: self.text_column, config.label_column: self.label_column}
            )
        else:
            self.config.do_eval = False

        self.data_collator = DataCollatorForTriples(self.tokenizer)
        self._run_config = {}

    def train(self) -> None:
        self.init_wandb()

        logger.info(f"train_dataset size: {len(self.train_dataset)*(self.config.subsample_train or 1)}")
        if self.config.do_dev_eval:
            logger.info(f"dev_dataset size: {len(self.val_dataset)}")
        if self.config.do_eval:
            logger.info(f"eval_dataset size: {len(self.eval_dataset)*(self.config.subsample_eval or 1)}")

        sampling_dataset = TripleSamplerDataset(
            triples_sampler_cls=self.triples_sampler_cls,
            config=self.config,
            text_column=self.config.text_column,
            label_column=self.config.label_column,
        )

        resample_callback = ResampleCallback(
            train_dataset=self.train_dataset,
            indexer=self.indexer,
            searcher=self.searcher_sampling,
            config=self.config,
        )

        logger.info("initial subsampling train dataset")
        train_subsample = subsample(
            self.train_dataset,
            self.config.subsample_train,
            self.config.stratify_splits,
            self.config.label_column,
        )

        logger.info("initial build index")
        initial_index_path = os.path.join(self.training_args.output_dir, f"checkpoint-{0}", "index")
        self.indexer.index(initial_index_path, train_subsample[self.config.text_column], gpus=True)

        sampling_dataset.create_sampling_data(
            dataset=train_subsample,
            searcher=self.searcher_sampling,
            tokenizer=self.tokenizer,
            index_path=initial_index_path,
        )

        if self.config.freeze_base_model:
            logger.info("Freezing base model parameters.")
            self.model.freeze_base_model()

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info(f"Training {n_params} parameters")

        trainer = TrainerWithEvaluation(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            train_dataset=sampling_dataset,
            eval_dataset=self.val_dataset if self.config.do_dev_eval else None,
            data_collator=self.data_collator,
            callbacks=[resample_callback],
            evaluate_fn=self.evaluate_fn,
            searcher=self.searcher_eval,
            eval_ks=self.config.eval_ks,
            text_column=self.text_column,
            label_column=self.label_column,
        )

        logger.info(f"Starting Trainer")
        with memory_stats():
            trainer.train()

        logger.info("Moving model to cpu and clearing cache.")
        self.model.cpu()
        torch.cuda.empty_cache()

        # cleanup
        for path in os.listdir(self.training_args.output_dir):
            index_path_to_delete = os.path.join(self.training_args.output_dir, path, "index")
            if path.startswith("checkpoint") and os.path.exists(index_path_to_delete):
                logger.info(f"[cleanup] deleting index at checkpoint {index_path_to_delete}")
                shutil.rmtree(index_path_to_delete)

        latest_index = os.path.join(self.training_args.output_dir, "index")
        train_subsample = subsample(
            self.train_dataset, self.config.final_subsample_train, self.config.stratify_splits, self.label_column
        )
        self.indexer.index(latest_index, train_subsample[self.text_column], gpus=True)
        train_subsample.save_to_disk(os.path.join(latest_index, "index_dataset"))

        if self.config.do_eval:
            eval_dataset = subsample(
                self.eval_dataset, self.config.subsample_eval, self.config.stratify_splits, self.label_column
            )
            metrics = trainer.evaluate(eval_dataset, index_path=latest_index, metric_key_prefix="test")
            logger.info(metrics)

        self.finish()

    # *** wandb stuff ***

    @property
    def run_config(self) -> dict:
        return self._run_config

    @run_config.setter
    def run_config(self, config: dict) -> None:
        self._run_config = config

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
