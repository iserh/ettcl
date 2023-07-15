import functools
from logging import getLogger

from datasets import Dataset
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments

from ettcl.core.config import RerankMLCTrainerConfig, RerankTrainerConfig
from ettcl.core.evaluate_mlc import evaluate_mlc
from ettcl.core.reranking import RerankTrainer
from ettcl.core.triple_sampling import TripleSamplingDataBuilderMLC
from ettcl.encoding import Encoder
from ettcl.indexing import Indexer
from ettcl.searching import Searcher

logger = getLogger(__name__)


class RerankMLCTrainer(RerankTrainer):
    config_cls = RerankMLCTrainerConfig
    triples_sampler_cls = TripleSamplingDataBuilderMLC
    evaluate_fn = staticmethod(evaluate_mlc)

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
        super().__init__(
            model,
            encoder,
            tokenizer,
            config,
            training_args,
            train_dataset,
            indexer,
            searcher_eval,
            val_dataset,
            eval_dataset,
            searcher_sampling,
        )
        self.config: RerankMLCTrainerConfig = self.config
        self.evaluate_fn = functools.partial(self.evaluate_fn, mlknn_s=self.config.mlknn_s, mlknn_k=self.config.mlknn_k)
