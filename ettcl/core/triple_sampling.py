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

Triple = tuple[int, ...]

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
        self.dataset = dataset_with_sampling_pids.remove_columns(self.triple_columns)
        self.nway = nway

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        triple = sample_triple(**self.sampling_data, idx=idx, nway=self.nway)
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


class TripleSamplingData:
    def __init__(
        self,
        passage_labels: Sequence[int],
        sampling_method: SamplingMethod = "random",
        probability_type: ProbabilityType = "uniform",
        nway: int = 2,
        sample_triples: bool = False,
        fill_missing: int | None = None,
    ) -> None:
        # self.searcher = searcher
        self.nway = nway
        self.fill_missing = fill_missing or nway - 1
        self.sampling_method = sampling_method
        self.sample_triples = sample_triples
        self.probability_type = probability_type
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

    def sampling_data_from_matches(
        self, match_pids: np.ndarray, match_scores: np.ndarray, label: int, idx: int | None = None
    ) -> dict[str, np.ndarray]:
        """Does not work batched!"""
        # compute a mask of which matched passages have also the same label
        label_pid_mask = np.isin(match_pids, self.pids_for_label[label], assume_unique=True)
        # matched pids that have the same label
        positive_pids = match_pids[label_pid_mask]
        positive_scores = match_scores[label_pid_mask]
        # matched pids that have different label
        negative_pids = match_pids[~label_pid_mask]
        negative_scores = match_scores[~label_pid_mask]

        if len(positive_pids) == 0:
            logger.warning(
                f"No passage with the same label ({label=}) was matched in the retrieval process. "
                f"Using a random sample of {self.fill_missing=} instead."
            )
            positive_pids = np.random.choice(self.pids_for_label[label], size=self.fill_missing)
            positive_scores = None

        if len(negative_pids) < self.nway - 1:
            logger.warning(
                f"Not enough passages with different label ({label=}) was matched in the retrieval process "
                f"(found only {len(negative_pids)}, but need at least {self.nway - 1}). "
                f"Using a random sample to fill up to {self.fill_missing=}."
            )
            # fill pids with other pids that have different labels
            pids_with_different_label = np.concatenate(
                [self.pids_for_label[l] for l in self.unique_labels if l != label]
            )
            non_retrieved_negatives = pids_with_different_label[
                ~np.isin(pids_with_different_label, negative_pids, assume_unique=True)
            ]
            negative_fill_pids = np.random.choice(non_retrieved_negatives, size=self.fill_missing - len(negative_pids))
            negative_pids = np.concatenate([negative_pids, negative_fill_pids])
            # fill scores with zeros
            fill_scores = np.zeros(len(negative_fill_pids))
            negative_scores = np.concatenate([negative_scores, fill_scores])

        match self.probability_type:
            case "scores":
                # use similarity scores as probabilities
                positive_probs = positive_scores
                negative_probs = negative_scores
            case "ranks":
                # use ranking as probabilities
                positive_probs = np.arange(len(positive_pids), 0, -1) if len(positive_pids) > 1 else None
                negative_probs = np.arange(len(negative_pids), 0, -1) if len(negative_pids) > 1 else None
            case "uniform":
                positive_probs = None
                negative_probs = None

        # normalize probs
        if positive_probs is not None:
            positive_probs = normalize_probs(positive_probs)
        if negative_probs is not None:
            negative_probs = normalize_probs(negative_probs)

        return {
            "positive_pids": positive_pids,
            "negative_pids": positive_pids,
            "positive_probs": positive_pids,
            "negative_probs": positive_pids,
        }

    def sampling_data_random(self, label: int, idx: int | None = None) -> dict[str, np.ndarray]:
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
        positive_pids: np.ndarray,
        negative_pids: np.ndarray,
        positive_probs: np.ndarray,
        negative_probs: np.ndarray,
        idx: int,
    ) -> dict[str, Triple]:
        triple = sample_triple(positive_pids, negative_pids, positive_probs, negative_probs, idx, self.nway)
        return {"triple": triple}


def sample_triple(
    positive_pids: np.ndarray,
    negative_pids: np.ndarray,
    positive_probs: np.ndarray,
    negative_probs: np.ndarray,
    idx: int,
    nway: int,
) -> dict[str, Triple]:
    sampled_positive = np.random.choice(positive_pids, size=1, p=positive_probs)
    sampled_negatives = np.random.choice(negative_pids, size=nway - 1, p=negative_probs)
    return (idx, sampled_positive, *sampled_negatives)


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    probs = scaler.fit_transform(probs.reshape(-1, 1)).reshape(-1)
    if probs.sum() == 0:
        return None
    else:
        return probs / probs.sum()
