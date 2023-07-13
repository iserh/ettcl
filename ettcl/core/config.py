from dataclasses import dataclass, field
from enum import Enum
from os import PathLike


class ProbabilityType(str, Enum):
    scores = "scores"
    ranks = "ranks"
    uniform = "uniform"


class SamplingMethod(str, Enum):
    random = "random"
    class_wise_random = "class_wise_random"
    searched = "searched"


@dataclass
class SamplingConfig:
    positive_always_random: bool = False
    lower_ranked_positives: bool = False
    searcher_sampling_k: int | None = 256
    sampling_method: SamplingMethod | str = "random"
    probability_type: ProbabilityType | str = "uniform"
    nway: int = 3


@dataclass
class RerankTrainerConfig(SamplingConfig):
    project: str | None = None
    do_dev_eval: bool = True
    dev_split_size: int | float = 0.1
    do_eval: bool = True
    eval_ks: tuple[int] = (1,)
    resample_interval: int | None = 1
    eval_interval: int | None = None
    dev_eval_interval: int | None = 1
    subsample_train: int | None = None
    subsample_eval: int | None = None
    final_subsample_train: int | None = None
    label_column: str = "label"
    text_column: str = "text"
    remove_columns: list[str] = field(default_factory=list)
    freeze_base_model: bool = False
    n_positives: int | None = None
    n_negatives: int | None = None
    log_model_artifact: bool = False
    stratify_splits: bool = True


@dataclass
class RerankMLCTrainerConfig(RerankTrainerConfig):
    label_column: str = "labels"
    stratify_splits: bool = False
    mlknn_s: float = 1
    mlknn_k: int = 10


@dataclass
class EvaluatorConfig:
    output_dir: str | PathLike = "evaluations"
    project: str | None = None
    eval_ks: tuple[int] = (1,)
    subsample_train: int | None = None
    subsample_eval: int | None = None
    label_column: str = "label"
    text_column: str = "text"
    prefix: str = "test"
    stratify_splits: bool = True


@dataclass
class EvaluatorMLCConfig(EvaluatorConfig):
    label_column: str = "labels"
    stratify_splits: bool = False
    mlknn_s: float = 1
    mlknn_k: int = 10
