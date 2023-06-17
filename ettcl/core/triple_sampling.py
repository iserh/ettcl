from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from ettcl.searching import BaseSearcher

logger = getLogger(__name__)


class TripleSampler:
    def __init__(self, dataset_with_triples: Dataset) -> None:
        self.triples = dataset_with_triples.select_columns("triple")
        self.dataset = dataset_with_triples.remove_columns("triple")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        triple = self.triples[idx]["triple"]
        features = self.dataset[triple]
        # unfold features
        return [{k: v[i] for k, v in features.items()} for i in range(len(triple))]


@dataclass
class DataCollatorForTriples(DataCollatorWithPadding):
    def __call__(self, features: list[list[dict[str, torch.Tensor]]]) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        n_way = len(features[0]) - 1

        # concatenate features
        flattened_features = sum(features, [])
        # default collate with padding
        batch = super().__call__(flattened_features)
        # unflatten batch
        batch = {k: v.view(batch_size, n_way + 1, *v.shape[1:]) for k, v in batch.items()}
        # label = 0, first passage is the positive one
        batch["labels"] = torch.zeros((batch_size,), dtype=torch.int64)

        return batch


class SamplingMode(str, Enum):
    scores = "scores"
    ranks = "ranks"
    uniform = "uniform"


def create_triples_random(
    dataset: Dataset,
    nway: int = 2,
) -> Dataset:
    # holds corresponding label for all passages in dataset
    passage_labels = np.array(dataset["label"], dtype=np.int64)
    # compute unique labels
    unique_labels = np.unique(passage_labels)
    # for each class holds pids that have that class assigned as label
    label_pids = {l: np.where(passage_labels == l)[0] for l in unique_labels}

    triples = []
    # iterate over all passages, together with their retrieved similar pids, scores and label
    for pid, label in enumerate(tqdm(passage_labels)):
        # pids that have the same label
        positive_pids = label_pids[label]
        # pids that have different label
        negative_pids = np.concatenate([label_pids[l] for l in unique_labels if l != label])

        positive_sample = np.random.choice(positive_pids, size=1)
        negative_samples = np.random.choice(negative_pids, size=nway - 1)

        triples.append((pid, *positive_sample.tolist(), *negative_samples.tolist()))

    return dataset.add_column("triple", triples)


def create_triples_ranked(
    dataset: Dataset,
    searcher: BaseSearcher,
    k: int,
    sampling_mode: SamplingMode = "uniform",
    nway: int = 2,
) -> Dataset:
    # holds corresponding label for all passages in dataset
    passage_labels = np.array(dataset["label"], dtype=np.int64)
    # for each class holds pids that have that class assigned as label
    label_pids = {l: np.where(passage_labels == l)[0] for l in np.unique(passage_labels)}

    # for all passages search for similar passages
    match_indices, match_scores = searcher.search(
        dataset["text"], k=k, return_tensors="np", progress_bar=True
    )

    triples = []
    # iterate over all passages, together with their retrieved similar pids, scores and label
    for pid, (indices, scores, label) in enumerate(
        tqdm(zip(match_indices, match_scores, passage_labels), total=len(passage_labels))
    ):
        # compute a mask of which matched passages have also the same label
        label_pid_mask = np.isin(indices, label_pids[label])
        # matched pids that have the same label
        positive_pids = indices[label_pid_mask]
        # matched pids that have different label
        negative_pids = indices[~label_pid_mask]

        if len(positive_pids) == 0:
            logger.warning(
                "No passage with the same label was matched in the retrieval process. Using a random sampled one instead."
            )
            positive_pids = np.random.choice(label_pids[label], size=1)

        match sampling_mode:
            case SamplingMode.scores:
                # use similarity scores as probabilities
                positive_probs = scores[label_pid_mask] if len(positive_pids) > 1 else None
                negative_probs = scores[~label_pid_mask] if len(negative_pids) > 1 else None
            case SamplingMode.ranks:
                # use ranking as probabilities
                positive_probs = (
                    np.arange(len(positive_pids), 0, -1) if len(positive_pids) > 1 else None
                )
                negative_probs = (
                    np.arange(len(negative_pids), 0, -1) if len(negative_pids) > 1 else None
                )
            case SamplingMode.uniform:
                positive_probs = None
                negative_probs = None

        # normalize probs
        if positive_probs is not None:
            positive_probs = normalize_probs(positive_probs)
        if negative_probs is not None:
            negative_probs = normalize_probs(negative_probs)

        positive_sample = np.random.choice(positive_pids, size=1, p=positive_probs)
        negative_samples = np.random.choice(negative_pids, size=nway - 1, p=negative_probs)

        triples.append((pid, *positive_sample.tolist(), *negative_samples.tolist()))

    return dataset.add_column("triple", triples)


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    probs = scaler.fit_transform(probs.reshape(-1, 1)).reshape(-1)
    if probs.sum() == 0:
        return None
    else:
        return probs / probs.sum()
