import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding

from ettcl.searching import Searcher
from ettcl.utils.multiprocessing import run_multiprocessed

logger = getLogger(__name__)


class TripleLoaderDataset:
    """Creates triples from triple indices already in dataset."""

    triple_column = "triple"

    def __init__(self, dataset_with_triples: Dataset) -> None:
        self.triples = dataset_with_triples.select_columns(self.triple_column)
        self.dataset = dataset_with_triples.remove_columns(self.triple_column)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        triple = self.triples[idx]["triple"]
        features = self.dataset[triple]
        # unfold features
        return [{k: v[i] for k, v in features.items()} for i in range(len(triple))]


class TripleSamplerDataset:
    """Samples triples online from sampling data."""

    triple_columns = ["positive_pids", "negative_pids", "positive_probs", "negative_probs"]

    def __init__(self, dataset_with_sampling_pids: Dataset, nway: int = 2) -> None:
        """Requires columns `"""
        self.sampling_data = dataset_with_sampling_pids.select_columns(self.triple_columns)
        self.sampling_data.set_format("pt")
        self.dataset = dataset_with_sampling_pids.remove_columns(self.triple_columns)
        self.dataset.set_format("pt")
        self.nway = nway

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        triple = sample_triple(**self.sampling_data[idx], idx=idx, nway=self.nway)
        features = self.dataset[triple]
        # unfold features
        return [{k: v[i] for k, v in features.items()} for i in range(len(triple))]


@dataclass
class DataCollatorForTriples(DataCollatorWithPadding):
    def __call__(self, features: list[list[dict[str, torch.Tensor]]]) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        n_way = len(features[0]) - 1

        # concatenate features
        flattened_features = list(itertools.chain(*features))
        # default collate with padding
        batch = super().__call__(flattened_features)
        # unflatten batch
        batch = {k: v.view(batch_size, n_way + 1, *v.shape[1:]) for k, v in batch.items()}
        # label = 0, first passage is the positive one
        batch["labels"] = torch.zeros((batch_size,), dtype=torch.int64)

        return batch


class ProbabilityType(str, Enum):
    scores = "scores"
    ranks = "ranks"
    uniform = "uniform"


class SamplingMethod(str, Enum):
    random = "random"
    searched = "searched"


class TripleSamplingDataBuilder:
    def __init__(
        self,
        passage_labels: Sequence[int],
        sampling_method: SamplingMethod = "random",
        probability_type: ProbabilityType = "uniform",
        nway: int = 2,
        sample_triples: bool = False,
        fill_missing: int | None = None,
        return_missing: bool = False,
    ) -> None:
        # self.searcher = searcher
        self.nway = nway
        self.fill_missing = fill_missing or nway - 1
        self.sampling_method = sampling_method
        self.sample_triples = sample_triples
        self.probability_type = probability_type
        self.return_missing = return_missing and sampling_method in [SamplingMethod.searched]
        # compute unique labels
        self.unique_labels = np.unique(passage_labels)
        # for each class holds pids that have that class assigned as label
        self.pids_for_label = {l: np.where(passage_labels == l)[0] for l in self.unique_labels}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        match self.sampling_method:
            case SamplingMethod.random:
                sampling_data = self.sampling_data_random(*args, **kwargs)
            case SamplingMethod.searched:
                sampling_data = self.sampling_data_from_matches(*args, **kwargs)
            case _:
                raise NotImplementedError(f"Sampling Method {self.sampling_method} not implemented.")

        if self.sample_triples:
            return self.triple_sample_from_sampling_data(sampling_data)
        else:
            return sampling_data

    @property
    def input_columns(self) -> list[str]:
        match self.sampling_method:
            case SamplingMethod.random:
                return ["label"]
            case SamplingMethod.searched:
                return ["match_pids", "match_scores", "label"]
            case _:
                raise NotImplementedError(f"Sampling Method {self.sampling_method} not implemented.")

    def sampling_data_from_matches(
        self, match_pids: np.ndarray, match_scores: np.ndarray, label: int, idx: int | None = None, *args, **kwargs
    ) -> dict[str, np.ndarray]:
        """Does not work batched!"""
        match_pids = np.array(match_pids)
        match_scores = np.array(match_scores)
        # compute a mask of which matched passages have also the same label
        label_pid_mask = np.isin(match_pids, self.pids_for_label[label], assume_unique=True)
        # matched pids that have the same label
        positive_pids = match_pids[label_pid_mask]
        positive_scores = match_scores[label_pid_mask]
        # matched pids that have different label
        negative_pids = match_pids[~label_pid_mask]
        negative_scores = match_scores[~label_pid_mask]

        missing_pos = len(positive_pids) < 1
        if missing_pos:
            # logger.warning(
            #     f"No passage with the same label ({label=}) was matched in the retrieval process. "
            #     f"Using a random sample of {self.fill_missing=} instead."
            # )
            positive_pids = np.random.choice(self.pids_for_label[label], size=self.fill_missing)
            positive_scores = None

        missing_neg = max(0, self.fill_missing - len(negative_pids))
        if missing_neg:
            # logger.warning(
            #     f"Not enough passages with different label ({label=}) was matched in the retrieval process "
            #     f"(found only {len(negative_pids)}, but need at least {self.fill_missing}). "
            #     f"Using {n_missing} random samples."
            # )
            # fill pids with other pids that have different labels
            pids_with_different_label = np.concatenate(
                [self.pids_for_label[l] for l in self.unique_labels if l != label]
            )
            non_retrieved_negatives = pids_with_different_label[
                ~np.isin(pids_with_different_label, negative_pids, assume_unique=True)
            ]
            negative_fill_pids = np.random.choice(non_retrieved_negatives, size=missing_neg)
            negative_pids = np.concatenate([negative_pids, negative_fill_pids])
            # fill scores with zeros
            fill_scores = np.full(missing_neg, fill_value=negative_scores.min() if len(negative_scores) else 1)
            negative_scores = np.concatenate([negative_scores, fill_scores])

        match self.probability_type:
            case "scores":
                # use similarity scores as probabilities
                positive_probs = positive_scores
                negative_probs = negative_scores
            case "ranks":
                # use ranking as probabilities
                positive_probs = (
                    np.arange(len(positive_pids), 0, -1).astype(np.float32) if len(positive_pids) > 1 else None
                )
                negative_probs = (
                    np.arange(len(negative_pids), 0, -1).astype(np.float32) if len(negative_pids) > 1 else None
                )
            case "uniform":
                positive_probs = None
                negative_probs = None

        positive_probs = normalize_probs(positive_probs) if positive_probs is not None else None
        negative_probs = normalize_probs(negative_probs) if negative_probs is not None else None

        sampling_data = {
            "positive_pids": positive_pids.tolist(),
            "negative_pids": negative_pids.tolist(),
            "positive_probs": positive_probs.tolist() if positive_probs is not None else None,
            "negative_probs": negative_probs.tolist() if negative_probs is not None else None,
        }

        if self.return_missing:
            sampling_data.update({"missing_pos": int(missing_pos)})
            sampling_data.update({"missing_neg": missing_neg})

        return sampling_data

    def sampling_data_random(self, label: int, idx: int | None = None, *args, **kwargs) -> dict[str, np.ndarray]:
        # pids that have the same label
        positive_pids = self.pids_for_label[label]
        # pids that have different label
        negative_pids = np.concatenate([self.pids_for_label[l] for l in self.unique_labels if l != label])

        return {
            "positive_pids": positive_pids,
            "negative_pids": negative_pids,
            "positive_probs": None,
            "negative_probs": None,
        }

    def triple_sample_from_sampling_data(
        self,
        positive_pids: torch.LongTensor,
        negative_pids: torch.LongTensor,
        positive_probs: torch.Tensor,
        negative_probs: torch.Tensor,
        idx: int,
    ) -> dict[str, torch.LongTensor :]:
        triple = sample_triple(positive_pids, negative_pids, positive_probs, negative_probs, idx, self.nway)
        return {"triple": triple}


def sample_triple(
    positive_pids: torch.LongTensor,
    negative_pids: torch.LongTensor,
    positive_probs: torch.Tensor,
    negative_probs: torch.Tensor,
    idx: int,
    nway: int,
) -> torch.LongTensor:
    if positive_probs is not None:
        positive_indices = positive_probs.multinomial(1)
    else:
        positive_indices = torch.randint(positive_pids.size(0), size=(1,))

    if negative_probs is not None:
        negative_indices = negative_probs.multinomial(nway - 1)
    else:
        negative_indices = torch.randint(negative_pids.size(0), size=(1,))

    sampled_positive = positive_pids[positive_indices]
    sampled_negatives = negative_pids[negative_indices]
    idx = torch.LongTensor([idx])

    return torch.cat([idx, sampled_positive, sampled_negatives])


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    probs = scaler.fit_transform(probs.reshape(-1, 1)).reshape(-1)
    if probs.sum() == 0:
        return None
    else:
        return probs
