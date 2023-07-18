from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput


def sentence_transformer_factory(model_name_or_path: str, pooling: str = 'mean') -> SentenceTransformer:
    transformer_model = Transformer(model_name_or_path)
    pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), pooling)
    return SentenceTransformer(modules=[transformer_model, pooling_model])


def sentence_transformer_for_reranking_factory(model_name_or_path: str, pooling: str = 'mean') -> SentenceTransformerForReranking:
    transformer_model = Transformer(model_name_or_path)
    pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), pooling)
    return SentenceTransformerForReranking(modules=[transformer_model, pooling_model])


@dataclass
class RerankingOutput(ModelOutput):
    loss: torch.Tensor = None
    scores: torch.Tensor = None


class SentenceTransformerForReranking(PreTrainedModel):
    base_model_prefix = "sentence_transformer.0.auto_model"
    _keys_to_ignore_on_load_missing = set(["default_labels"])
    _keys_to_ignore_on_save = ["default_labels"]

    def __init__(
        self,
        model_name_or_path: str | PathLike | None = None,
        modules: Iterable[nn.Module] | None = None,
        device: str | None = None,
        cache_folder: str | None = None,
        use_auth_token: bool | str | None = None,
    ):
        sentence_transformer = SentenceTransformer(
            model_name_or_path=model_name_or_path,
            modules=modules,
            device=device,
            cache_folder=cache_folder,
            use_auth_token=use_auth_token,
        )
        config = sentence_transformer._first_module().auto_model.config
        super().__init__(config)
        self.config = config
        self.sentence_transformer = sentence_transformer
        self.register_buffer("default_labels", torch.zeros(1, dtype=torch.int64))

    @property
    def ST(self) -> SentenceTransformer:
        return self.sentence_transformer

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        state_dict = {self.base_model_prefix + "." + k: tensor for k, tensor in state_dict.items()}
        return super().load_state_dict(state_dict, strict)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        target_scores: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> RerankingOutput | tuple[torch.Tensor, torch.Tensor]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # input has shape (BATCH, nway, input_length)
        batch_size = input_ids.shape[0]
        nway = input_ids.shape[1]
        input_length = input_ids.shape[2]

        features = self.sentence_transformer(
            dict(
                input_ids=input_ids.view(-1, input_length) if input_ids is not None else None,
                attention_mask=attention_mask.view(-1, input_length) if attention_mask is not None else None,
                token_type_ids=token_type_ids.view(-1, input_length) if token_type_ids is not None else None,
                return_dict=return_dict,
            )
        )

        embeddings = features["sentence_embedding"].view(batch_size, nway, -1)
        q = embeddings[:, :1]
        D = embeddings[:, 1:]

        scores = D @ q.permute(0, 2, 1)
        scores = scores.squeeze(2)

        if labels is None:
            # default labels <=> 0 for whole batch, which is selecting the first document as target for cross_entropy_loss
            labels = self.default_labels.broadcast_to(batch_size)

        loss = None
        if labels.shape == torch.Size([batch_size]):
            loss = F.cross_entropy(scores, labels)

        else:
            # if target_scores are provided (e.g. from a cross-encoder) knowledge is distilled by computing kl divergence of the ranking distributions
            assert labels.shape == torch.Size(
                [batch_size, nway - 1]
            ), f"Expected label scores to of shape {torch.Size([batch_size, nway - 1])} (bsize, nway-1), got {labels.shape}"

            target_scores = labels * self.config.distillation_alpha
            target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

            log_scores = F.log_softmax(scores, dim=-1)
            loss = F.kl_div(log_scores, target_scores, reduction="batchmean", log_target=True)

        # TODO: in-batch negatives loss

        if not return_dict:
            return (loss, scores)

        return RerankingOutput(
            loss=loss.unsqueeze(0),
            scores=scores,
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | PathLike | None, *model_args, **kwargs
    ) -> SentenceTransformerForReranking:
        return cls(pretrained_model_name_or_path, *model_args, **kwargs)

    def save_pretrained(self, save_directory: str | PathLike, *args, **kwargs):
        self.sentence_transformer.save(path=save_directory)
