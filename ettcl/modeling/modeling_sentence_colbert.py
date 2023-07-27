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
from ettcl.modeling.modeling_colbert import (
    ColBERTForReranking,
    ColBERTModel,
    ColbertOutputWitgCrossAttentions,
    ColbertRerankingOutput,
)

logger = getLogger(__name__)


class SentenceColBERTModel(ColBERTModel):
    def forward(
        self,
        input_ids: list[torch.Tensor],
        attention_mask: list[torch.Tensor],
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        num_sentences = input_ids.shape[1]
        seq_length = input_ids.shape[2]

        encoder_outputs = self.LM(
            input_ids.view(-1, seq_length),
            attention_mask=attention_mask.view(-1, seq_length),
            return_dict=return_dict,
            **kwargs,
        )

        sequence_output = encoder_outputs[0]
        compressed_output = self.linear(sequence_output) if self.linear is not None else sequence_output
        compressed_output = compressed_output * attention_mask.view(-1, seq_length, 1)

        compressed_output = compressed_output.view(batch_size, num_sentences, seq_length, -1)
        sentence_embeddings = compressed_output.mean(2)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=2)

        if not return_dict:
            return (sentence_embeddings,) + encoder_outputs[1:]

        return ColbertOutputWitgCrossAttentions(
            normalized_output=sentence_embeddings,
            output_mask=attention_mask.any(-1),
            hidden_states=encoder_outputs.hidden_states,
        )


class SentenceColBERTForReranking(ColBERTForReranking):
    _keys_to_ignore_on_load_missing = set(["default_labels"])
    _keys_to_ignore_on_save = set(["default_labels"])

    def __init__(
        self,
        config: ColBERTConfig,
        base_model_class: type[PreTrainedModel] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.colbert = SentenceColBERTModel(config, base_model_class)
        self.register_buffer("default_labels", torch.zeros(1, dtype=torch.int64))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        target_scores: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> ColbertRerankingOutput | tuple[torch.Tensor, torch.Tensor]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # input has shape (BATCH, nway, input_length)
        batch_size = input_ids.shape[0]
        nway = input_ids.shape[1]
        num_sentences = input_ids.shape[2]
        seq_length = input_ids.shape[3]

        outputs = self.colbert(
            input_ids=input_ids.view(-1, num_sentences, seq_length) if input_ids is not None else None,
            attention_mask=attention_mask.view(-1, num_sentences, seq_length) if attention_mask is not None else None,
            return_dict=return_dict,
        )

        scores, unreduced_scores, D_mask = self.compute_scores(
            sequence_output=outputs[0],
            attention_mask=attention_mask.any(-1),
            nway=nway,
            input_length=num_sentences,
        )

        loss = self.compute_loss(
            scores=scores,
            labels=labels,
            target_scores=target_scores,
            batch_size=batch_size,
            nway=nway,
        )

        if not return_dict:
            return (loss, scores, unreduced_scores)

        return ColbertRerankingOutput(
            loss=loss.unsqueeze(0),
            scores=scores,
            unreduced_scores=unreduced_scores,
            doc_mask=D_mask,
            hidden_states=outputs.hidden_states,
        )
