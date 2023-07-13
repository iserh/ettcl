#!/usr/bin/env python
from __future__ import annotations

import os
from argparse import ArgumentTypeError
from dataclasses import asdict, dataclass, field
from typing import Generic, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class Value(Generic[T]):
    value: T | None = None
    desc: str | None = None


@dataclass
class RunParams:
    dataset: Value[str]
    model: Value[str]
    seed: Value[int] = field(default_factory=lambda: Value(12345))
    num_sentences: Value[int] = field(default_factory=lambda: Value(16))
    model_config: Value[dict] = field(default_factory=lambda: Value(dict()))
    tokenizer: Value[dict] = field(default_factory=lambda: Value(dict()))
    indexer: Value[dict] = field(default_factory=lambda: Value(dict()))
    searcher_eval: Value[dict] = field(default_factory=lambda: Value(dict()))
    searcher_sampling: Value[dict] = field(default_factory=lambda: Value(dict()))
    training: Value[dict] = field(default_factory=lambda: Value(dict()))
    config: Value[dict] = field(default_factory=lambda: Value(dict()))

    @classmethod
    def from_yml(cls, path: str | os.PathLike) -> RunParams:
        with open(path, "r") as fp:
            d = yaml.load(fp, yaml.SafeLoader)

        return cls(
            **{
                k: Value(d[k]["value"], d[k].get("desc", None))
                for k in cls.__dataclass_fields__.keys()
                if d.get(k, None) is not None and d[k].get("value", None) is not None
            }
        )

    def to_dict(self) -> dict:
        return asdict(self)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    import importlib
    import warnings
    from argparse import ArgumentParser

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="resource_tracker: There appear to be .* leaked semaphore objects to clean up at shutdown",
    )

    parser = ArgumentParser()
    parser.add_argument("script", type=str)

    parser.add_argument("--file", type=str)  # load RunParams from file
    parser.add_argument("--log_level", type=str, default="INFO")

    # load RunParams from argparse
    parser.add_argument("--project", type=str)  # required
    parser.add_argument("--dataset", type=str)  # required
    parser.add_argument("--model", type=str)  # required

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num_sentences", type=int)

    parser.add_argument("--mconfig_compression_dim", type=int)

    parser.add_argument("--tokenizer_query_maxlen", type=int)
    parser.add_argument("--tokenizer_doc_maxlen", type=int)
    parser.add_argument("--tokenizer_query_token", type=str)
    parser.add_argument("--tokenizer_doc_token", type=str)
    parser.add_argument("--tokenizer_add_special_tokens", type=str2bool)

    parser.add_argument("--training_optim", type=str, default="adamw_torch")
    parser.add_argument("--training_log_level", type=str, default="info")
    parser.add_argument("--training_save_total_limit", type=int, default=1)
    parser.add_argument("--training_save_strategy", type=str, default="epoch")
    parser.add_argument("--training_save_steps", type=int, default=1)
    parser.add_argument("--training_logging_steps", type=int, default=100)
    parser.add_argument("--training_eval_steps", type=int, default=1)
    parser.add_argument("--training_evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--training_load_best_model_at_end", type=str2bool, default=True)
    parser.add_argument("--training_metric_for_best_model", type=str, default="f1/micro")
    parser.add_argument("--training_num_train_epochs", type=int, default=5)
    parser.add_argument("--training_per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--training_gradient_accumulation_steps", type=int)

    parser.add_argument("--indexer_nbits", type=int)
    parser.add_argument("--indexer_kmeans_niters", type=int)
    parser.add_argument("--indexer_num_partitions_fac", type=int)
    parser.add_argument("--indexer_string_factory", type=str)
    parser.add_argument("--indexer_train_size", type=int)
    parser.add_argument("--indexer_metric_type", type=str)

    parser.add_argument("--searcher_eval_ncells", type=int)
    parser.add_argument("--searcher_eval_centroid_score_threshold", type=float)
    parser.add_argument("--searcher_eval_ndocs", type=int)
    parser.add_argument("--searcher_eval_plaid_num_elem_batch", type=int)
    parser.add_argument("--searcher_eval_skip_plaid_stage_3", type=str2bool)
    parser.add_argument("--searcher_eval_plaid_stage_2_3_cpu", type=str2bool)

    parser.add_argument("--searcher_sampling_ncells", type=int)
    parser.add_argument("--searcher_sampling_centroid_score_threshold", type=float)
    parser.add_argument("--searcher_sampling_ndocs", type=int)
    parser.add_argument("--searcher_sampling_plaid_num_elem_batch", type=int)
    parser.add_argument("--searcher_sampling_skip_plaid_stage_3", type=str2bool)
    parser.add_argument("--searcher_sampling_plaid_stage_2_3_cpu", type=str2bool)

    parser.add_argument("--config_do_dev_eval", type=str2bool)
    parser.add_argument("--config_dev_split_size", type=float)
    parser.add_argument("--config_do_eval", type=str2bool)
    parser.add_argument("--config_eval_ks", type=int, nargs="+")
    parser.add_argument("--config_resample_interval", type=int)
    parser.add_argument("--config_eval_interval", type=int)
    parser.add_argument("--config_dev_eval_interval", type=int)
    parser.add_argument("--config_searcher_sampling_k", type=int)
    parser.add_argument("--config_subsample_train", type=float)
    parser.add_argument("--config_subsample_eval", type=float)
    parser.add_argument("--config_final_subsample_train", type=float)
    parser.add_argument("--config_label_column", type=str)
    parser.add_argument("--config_text_column", type=str)
    parser.add_argument("--config_remove_columns", type=str, nargs="+")
    parser.add_argument("--config_freeze_base_model", type=str2bool)
    parser.add_argument("--config_sampling_method", type=str)
    parser.add_argument("--config_probability_type", type=str)
    parser.add_argument("--config_nway", type=int)
    parser.add_argument("--config_n_positives", type=int)
    parser.add_argument("--config_n_negatives", type=int)
    parser.add_argument("--config_positive_always_random", type=str2bool)
    parser.add_argument("--config_lower_ranked_positives", type=str2bool)
    parser.add_argument("--config_log_model_artifact", type=str2bool)
    parser.add_argument("--config_stratify_splits", type=str2bool)
    parser.add_argument("--config_prefix", type=str)
    parser.add_argument("--config_mlknn_s", type=float)

    args = parser.parse_args()

    if args.file is not None:
        params = RunParams.from_yml(args.file)

    else:
        if args.project is None:
            raise ValueError("Specify --project")
        if args.dataset is None:
            raise ValueError("Specify --dataset")
        if args.model is None:
            raise ValueError("Specify --model")

        mconfig = {}
        tokenizer = {}
        training = {}
        indexer = {}
        searcher_eval = {}
        searcher_sampling = {}
        config = {"project": args.project}

        for name, value in args._get_kwargs():
            if name.startswith("mconfig_") and value is not None:
                mconfig[name.replace("mconfig_", "")] = value

            elif name.startswith("tokenizer_") and value is not None:
                tokenizer[name.replace("tokenizer_", "")] = value

            elif name.startswith("training_") and value is not None:
                training[name.replace("training_", "")] = value

            elif name.startswith("indexer_") and value is not None:
                indexer[name.replace("indexer_", "")] = value

            elif name.startswith("searcher_eval_") and value is not None:
                searcher_eval[name.replace("searcher_eval_", "")] = value

            elif name.startswith("searcher_sampling_") and value is not None:
                searcher_sampling[name.replace("searcher_sampling_", "")] = value

            elif name.startswith("config_") and value is not None:
                config[name.replace("config_", "")] = value

            else:
                pass

        params = RunParams(
            dataset=Value(args.dataset),
            model=Value(args.model),
            model_config=Value(mconfig),
            tokenizer=Value(tokenizer),
            indexer=Value(indexer),
            searcher_eval=Value(searcher_eval),
            searcher_sampling=Value(searcher_sampling),
            training=Value(training),
            config=Value(config),
        )

        if args.seed is not None:
            params.seed = Value(args.seed)

        if args.num_sentences is not None:
            params.num_sentences = Value(args.num_sentences)

    module_path = args.script
    module_name = "run_script"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main(params.to_dict(), args.log_level)
