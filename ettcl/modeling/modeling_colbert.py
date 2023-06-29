from __future__ import annotations

import os
from dataclasses import dataclass
from logging import getLogger
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto.auto_factory import _get_model_class

from ettcl.modeling.configuration_colbert import ColBERTConfig

logger = getLogger(__name__)


@dataclass
class ColbertOutputWitgCrossAttentions(ModelOutput):
    normalized_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ColbertRerankingOutput(ModelOutput):
    scores: torch.Tensor = None
    unreduced_scores: torch.Tensor = None
    loss: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class ColBERTPreTrainedModel(PreTrainedModel):
    config_class = ColBERTConfig
    base_model_prefix = "colbert"

    # these will be filled in the `from_pretrained` method
    _keys_to_ignore_on_load_missing = set()
    _keys_to_ignore_on_load_unexpected = set()
    _keys_to_ignore_on_save = set()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = ColBERTConfig.from_pretrained(pretrained_model_name_or_path)
        else:
            assert isinstance(config, PretrainedConfig)
            config = ColBERTConfig.from_dict(config.to_dict())

        base_config = AutoConfig.for_model(**config.to_dict())
        base_model_cls: type[PreTrainedModel] = _get_model_class(base_config, AutoModel._model_mapping)

        if base_model_cls._keys_to_ignore_on_load_missing is not None:
            cls._keys_to_ignore_on_load_missing.update(base_model_cls._keys_to_ignore_on_load_missing)

        if base_model_cls._keys_to_ignore_on_load_unexpected is not None:
            cls._keys_to_ignore_on_load_unexpected.update(base_model_cls._keys_to_ignore_on_load_unexpected)

        if base_model_cls._keys_to_ignore_on_save is not None:
            cls._keys_to_ignore_on_save.update(base_model_cls._keys_to_ignore_on_save)

        return super(ColBERTPreTrainedModel, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, config=config, **kwargs
        )


class ColBERTModel(ColBERTPreTrainedModel):
    def __init__(
        self,
        config: ColBERTConfig,
        base_model_class: type[PreTrainedModel] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.output_dimensionality = self.config.compression_dim or config.hidden_size

        # if base_model_class is provided this model class is used, otherwise revert to config definition
        if base_model_class is not None:
            base_model = base_model_class(config)
        else:
            # if config is a colbert config, we need to infer the config class from
            # the config_dict (which contains the `model_type` key)
            config = AutoConfig.for_model(**config.to_dict())
            base_model = AutoModel.from_config(config)

        # set the correct base_model_prefix
        self.lm_base_model_prefix = base_model.base_model_prefix
        # set base model as attribute
        setattr(self, self.lm_base_model_prefix, base_model)

        # add final compression layer if compression_dim is defined
        self.linear = (
            nn.Linear(config.hidden_size, self.config.compression_dim, bias=False)
            if self.config.compression_dim is not None
            else None
        )

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def LM(self) -> PreTrainedModel:
        return getattr(self, self.lm_base_model_prefix)

    def freeze_base_model(self) -> None:
        for param in self.LM.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.LM(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        compressed_output = self.linear(sequence_output) if self.linear is not None else sequence_output
        compressed_output = compressed_output * attention_mask.unsqueeze(-1)
        normalized_output = F.normalize(compressed_output, p=2, dim=2)

        if not return_dict:
            return (normalized_output,) + encoder_outputs[1:]

        return ColbertOutputWitgCrossAttentions(
            normalized_output=normalized_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class ColBERTForReranking(ColBERTPreTrainedModel):
    _keys_to_ignore_on_load_missing = set(["default_labels"])
    _keys_to_ignore_on_save = set(["default_labels"])

    def __init__(
        self,
        config: ColBERTConfig,
        base_model_class: type[PreTrainedModel] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.colbert = ColBERTModel(config, base_model_class)
        self.register_buffer("default_labels", torch.zeros(1, dtype=torch.int64))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        target_scores: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> ColbertRerankingOutput | tuple[torch.Tensor, torch.Tensor]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # input has shape (BATCH, nway+1, input_length)
        batch_size = input_ids.shape[0]
        nway = input_ids.shape[1]
        input_length = input_ids.shape[2]

        outputs = self.colbert(
            input_ids=input_ids.view(-1, input_length) if input_ids is not None else None,
            attention_mask=attention_mask.view(-1, input_length) if attention_mask is not None else None,
            token_type_ids=token_type_ids.view(-1, input_length) if token_type_ids is not None else None,
            return_dict=return_dict,
        )

        # outputs[0] holds normalized output, shape (BATCH * (nway+1), input_length, embedding_dim)
        embedding_dim = outputs[0].shape[-1]
        # reshape to (BATCH, nway+1, input_length, embedding_dim)
        sequence_output = outputs[0].view(-1, nway, input_length, embedding_dim)
        # select queries
        Q = sequence_output[:, 0]
        # select nway documents, view as (BATCH * nway, input_length, embedding_dim)
        D = sequence_output[:, 1:].reshape(-1, *sequence_output.shape[2:])
        D_mask = attention_mask[:, 1:].reshape(-1, *attention_mask.shape[2:])

        # Repeat each query encoding for every corresponding document, shape (BATCH * nway, input_length, embedding_dim)
        Q_duplicated = Q.repeat_interleave(nway - 1, dim=0).contiguous()

        # compute similarity scores between all embeddings using batched matrix-multiplication
        unreduced_scores = colbert_score(Q_duplicated, D)
        # reduce these scores via `MaxSim`
        scores = maxsim_reduction(unreduced_scores, D_mask).view(-1, nway - 1)

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
            return (scores, unreduced_scores)

        return ColbertRerankingOutput(
            scores=scores,
            unreduced_scores=unreduced_scores,
            loss=loss.unsqueeze(0),
            hidden_states=outputs.hidden_states,
        )


def colbert_score(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Q, D have to be l2 normalized!"""
    assert Q.dim() == 3, Q.size()
    assert D.dim() == 3, D.size()
    assert Q.shape[0] in [1, D.shape[0]]

    # use half precision on gpu
    if Q.device.type != "cpu":
        Q = Q.half()
        D = D.half()

    # exhaustive cosine similarity between all embeddings
    return D @ Q.permute(0, 2, 1)


def maxsim_reduction(scores: torch.Tensor, D_mask: torch.LongTensor) -> torch.Tensor:
    # we don't want to match padding embeddings of the docs (cosine_sim in [-1, +1])
    scores[~D_mask.bool()] = -2
    return scores.max(1).values.sum(-1)
