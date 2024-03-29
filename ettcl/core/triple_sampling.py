import itertools
import os
from collections.abc import Sequence
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from ettcl.core.config import ProbabilityType, SamplingConfig, SamplingMethod
from ettcl.core.search import search_dataset
from ettcl.indexing import Indexer
from ettcl.searching import Searcher

logger = getLogger(__name__)


class TripleSamplingDataBuilder:
    def __init__(
        self,
        passage_labels: Sequence[int],
        sampling_method: SamplingMethod = "random",
        probability_type: ProbabilityType = "uniform",
        nway: int = 3,
        n_positives: int | None = None,
        n_negatives: int | None = None,
        return_missing: bool = False,
        positive_always_random: bool = False,
        lower_ranked_positives: bool = False,
        sample_triples: bool = False,
    ) -> None:
        assert nway >= 3, "NWay tuples have to be at least of size 3 (anchor, positive, negative)"
        self.nway = nway
        self.sampling_method = sampling_method
        self.probability_type = probability_type
        self.n_positives = n_positives or 1
        self.n_negatives = n_negatives or nway - 2
        self.return_missing = return_missing and sampling_method in [SamplingMethod.searched]
        self.positive_always_random = positive_always_random
        self.lower_ranked_positives = lower_ranked_positives
        self.sample_triples = sample_triples
        # compute unique labels
        self.unique_labels = np.unique(passage_labels)
        self.num_labels = len(self.unique_labels)

        if self.sampling_method == SamplingMethod.class_wise_random:
            assert (nway - 2) % (self.num_labels - 1) == 0, "nway must be 2 + n*(num_labels-1)"

        # for each class holds pids that have that class assigned as label
        self.pids_for_label = {l: np.where(passage_labels == l)[0] for l in self.unique_labels}
        self.scaler = MinMaxScaler()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        match self.sampling_method:
            case SamplingMethod.random:
                sampling_data = self.sampling_data_random(*args, **kwargs)
            case SamplingMethod.class_wise_random:
                sampling_data = self.sampling_data_cw_random(*args, **kwargs)
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
            case SamplingMethod.class_wise_random:
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

        pids_with_same_label = self.pids_for_label[label]
        pids_with_different_label = np.concatenate([self.pids_for_label[l] for l in self.unique_labels if l != label])
        # remove self from positive list
        m = match_pids != idx
        match_pids = match_pids[m]
        match_scores = match_scores[m]
        pids_with_same_label = pids_with_same_label[pids_with_same_label != idx]
        same_label_mask = np.isin(match_pids, pids_with_same_label, assume_unique=True)

        assert len(pids_with_same_label), f"There exist only 1 pid with label {label}, check your data preprocessing."

        n_positives = min(self.n_positives, len(pids_with_same_label))
        n_negatives = min(self.n_negatives, len(pids_with_different_label))

        if self.positive_always_random:
            positive_pids = pids_with_same_label
            positive_scores = None
        else:
            # matched pids that have the same label
            positive_pids = match_pids[same_label_mask]
            positive_scores = match_scores[same_label_mask]

        if missing_pos := max(0, n_positives - len(positive_pids)):
            non_retrieved_positives = pids_with_same_label[
                ~np.isin(pids_with_same_label, positive_pids, assume_unique=True)
            ]
            positive_fill_pids = np.random.choice(non_retrieved_positives, size=missing_pos, replace=len(non_retrieved_positives) < missing_pos)
            positive_pids = np.concatenate([positive_pids, positive_fill_pids])
            fill_scores = np.full(missing_pos, fill_value=positive_scores.min() if len(positive_scores) else 1)
            positive_scores = np.concatenate([positive_scores, fill_scores])

        # matched pids that have different label
        negative_pids = match_pids[~same_label_mask]
        negative_scores = match_scores[~same_label_mask]

        if missing_neg := max(0, n_negatives - len(negative_pids)):
            non_retrieved_negatives = pids_with_different_label[
                ~np.isin(pids_with_different_label, negative_pids, assume_unique=True)
            ]
            negative_fill_pids = np.random.choice(non_retrieved_negatives, size=missing_neg)
            negative_pids = np.concatenate([negative_pids, negative_fill_pids])
            fill_scores = np.full(missing_neg, fill_value=negative_scores.min() if len(negative_scores) else 1)
            negative_scores = np.concatenate([negative_scores, fill_scores])

        match self.probability_type:
            case "scores":
                # use similarity scores as probabilities
                positive_probs = positive_scores
                negative_probs = negative_scores

            case "ranks":
                # use ranking as probabilities
                if not self.positive_always_random:
                    positive_probs = (
                        np.arange(len(positive_pids), 0, -1).astype(np.float32) if len(positive_pids) > 1 else None
                    )
                    if self.lower_ranked_positives:
                        positive_probs = np.flip(positive_probs)
                negative_probs = (
                    np.arange(len(negative_pids), 0, -1).astype(np.float32) if len(negative_pids) > 1 else None
                )

            case "uniform":
                positive_probs = None
                negative_probs = None

            case _:
                raise NotImplementedError(self.probability_type)

        positive_probs = normalize_probs(positive_probs, self.scaler) if positive_probs is not None else None
        negative_probs = normalize_probs(negative_probs, self.scaler) if negative_probs is not None else None

        sampling_data = {
            "positive_pids": positive_pids.tolist(),
            "negative_pids": negative_pids.tolist(),
            "positive_probs": positive_probs.tolist()
            if positive_probs is not None
            else np.ones_like(positive_pids, dtype=np.float32).tolist(),
            "negative_probs": negative_probs.tolist()
            if negative_probs is not None
            else np.ones_like(negative_pids, dtype=np.float32).tolist(),
        }

        if self.return_missing:
            assert isinstance(missing_pos, int), "Value must be int"
            assert isinstance(missing_neg, int), "Value must be int"
            sampling_data.update({"missing_pos": missing_pos})
            sampling_data.update({"missing_neg": missing_neg})

        return sampling_data

    def sampling_data_random(self, label: int, idx: int | None = None, *args, **kwargs) -> dict[str, np.ndarray]:
        # pids that have the same label
        positive_pids = self.pids_for_label[label]
        positive_pids = positive_pids[positive_pids != idx]
        # pids that have different label
        negative_pids = np.concatenate([self.pids_for_label[l] for l in self.unique_labels if l != label])

        assert len(positive_pids), f"There exist only 1 pid with label {label}, check your data preprocessing."

        return {
            "positive_pids": positive_pids.tolist(),
            "negative_pids": negative_pids.tolist(),
            "positive_probs": np.ones_like(positive_pids, dtype=np.float32).tolist(),
            "negative_probs": np.ones_like(negative_pids, dtype=np.float32).tolist(),
        }

    def sampling_data_cw_random(self, label: int, idx: int | None = None, *args, **kwargs) -> dict[str, np.ndarray]:
        n = (self.nway - 2) // (self.num_labels)
        positive_pids = self.pids_for_label[label]
        negative_pids = np.concatenate([np.random.choice(self.pids_for_label[l], size=n) for l in self.unique_labels])

        return {
            "positive_pids": positive_pids.tolist(),
            "negative_pids": negative_pids.tolist(),
            "positive_probs": np.ones_like(positive_pids, dtype=np.float32).tolist(),
            "negative_probs": np.ones_like(negative_pids, dtype=np.float32).tolist(),
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


class TripleSamplingDataBuilderMLC(TripleSamplingDataBuilder):
    def __init__(
        self,
        passage_labels: Sequence[int],
        sampling_method: SamplingMethod = "random",
        probability_type: ProbabilityType = "uniform",
        nway: int = 3,
        n_positives: int | None = None,
        n_negatives: int | None = None,
        return_missing: bool = False,
        positive_always_random: bool = False,
        lower_ranked_positives: bool = False,
        sample_triples: bool = False,
    ) -> None:
        assert nway >= 3, "NWay tuples have to be at least of size 3 (anchor, positive, negative)"
        self.nway = nway
        self.sampling_method = sampling_method
        self.probability_type = probability_type
        self.n_positives = n_positives or 1
        self.n_negatives = n_negatives or nway - 2
        self.return_missing = return_missing and sampling_method in [SamplingMethod.searched]
        self.positive_always_random = positive_always_random
        self.lower_ranked_positives = lower_ranked_positives
        self.sample_triples = sample_triples

        self.unique_labels = np.unique(np.concatenate(passage_labels))
        self.num_labels = len(self.unique_labels)

        if self.sampling_method == SamplingMethod.class_wise_random:
            assert (nway - 2) % (self.num_labels - 1) == 0, "nway must be 2 + n*(num_labels-1)"

        # for each class holds pids that have that class assigned as label
        k = max([len(labels) for labels in passage_labels])
        labels_padded = np.stack([np.pad(labels, (0, k - len(labels))) for labels in passage_labels])
        self.pids_for_label = {l: np.where(labels_padded == l)[0] for l in self.unique_labels}
        self.scaler = MinMaxScaler()

    @property
    def input_columns(self) -> list[str]:
        match self.sampling_method:
            case SamplingMethod.random:
                return ["labels"]
            case SamplingMethod.class_wise_random:
                return ["labels"]
            case SamplingMethod.searched:
                return ["match_pids", "match_scores", "labels"]
            case _:
                raise NotImplementedError(f"Sampling Method {self.sampling_method} not implemented.")

    def sampling_data_from_matches(
        self,
        match_pids: np.ndarray,
        match_scores: np.ndarray,
        labels: list[int],
        idx: int | None = None,
        *args,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Does not work batched!"""
        match_pids = np.array(match_pids)
        match_scores = np.array(match_scores)

        pids_with_same_label = np.unique(np.concatenate([self.pids_for_label[l] for l in labels]))
        pids_with_different_label = np.unique(
            np.concatenate([self.pids_for_label[l] for l in self.unique_labels if l not in labels])
        )
        # remove self from positive list
        m = match_pids != idx
        match_pids = match_pids[m]
        match_scores = match_scores[m]
        pids_with_same_label = pids_with_same_label[pids_with_same_label != idx]
        same_label_mask = np.isin(match_pids, pids_with_same_label, assume_unique=True)

        if not len(pids_with_same_label):
            pids_with_same_label = np.array([idx])
            logger.warning(f"No positive pid with labels {labels} existent.")

        n_positives = min(self.n_positives, len(pids_with_same_label))
        n_negatives = min(self.n_negatives, len(pids_with_different_label))

        if self.positive_always_random:
            positive_pids = pids_with_same_label
            positive_scores = None
        else:
            # matched pids that have the same label
            positive_pids = match_pids[same_label_mask]
            positive_scores = match_scores[same_label_mask]

        if missing_pos := max(0, n_positives - len(positive_pids)):
            non_retrieved_positives = pids_with_same_label[
                ~np.isin(pids_with_same_label, positive_pids, assume_unique=True)
            ]
            positive_fill_pids = np.random.choice(non_retrieved_positives, size=missing_pos)
            positive_pids = np.concatenate([positive_pids, positive_fill_pids])
            fill_scores = np.full(missing_pos, fill_value=positive_scores.min() if len(positive_scores) else 1)
            positive_scores = np.concatenate([positive_scores, fill_scores])

        # matched pids that have different label
        negative_pids = match_pids[~same_label_mask]
        negative_scores = match_scores[~same_label_mask]

        if missing_neg := max(0, n_negatives - len(negative_pids)):
            non_retrieved_negatives = pids_with_different_label[
                ~np.isin(pids_with_different_label, negative_pids, assume_unique=True)
            ]
            negative_fill_pids = np.random.choice(non_retrieved_negatives, size=missing_neg)
            negative_pids = np.concatenate([negative_pids, negative_fill_pids])
            fill_scores = np.full(missing_neg, fill_value=negative_scores.min() if len(negative_scores) else 1)
            negative_scores = np.concatenate([negative_scores, fill_scores])

        match self.probability_type:
            case "scores":
                # use similarity scores as probabilities
                positive_probs = positive_scores
                negative_probs = negative_scores

            case "ranks":
                # use ranking as probabilities
                if not self.positive_always_random:
                    positive_probs = (
                        np.arange(len(positive_pids), 0, -1).astype(np.float32) if len(positive_pids) > 1 else None
                    )
                    if self.lower_ranked_positives:
                        positive_probs = np.flip(positive_probs)
                negative_probs = (
                    np.arange(len(negative_pids), 0, -1).astype(np.float32) if len(negative_pids) > 1 else None
                )

            case "uniform":
                positive_probs = None
                negative_probs = None

            case _:
                raise NotImplementedError(self.probability_type)

        positive_probs = normalize_probs(positive_probs, self.scaler) if positive_probs is not None else None
        negative_probs = normalize_probs(negative_probs, self.scaler) if negative_probs is not None else None

        sampling_data = {
            "positive_pids": positive_pids.tolist(),
            "negative_pids": negative_pids.tolist(),
            "positive_probs": positive_probs.tolist()
            if positive_probs is not None
            else np.ones_like(positive_pids, dtype=np.float32).tolist(),
            "negative_probs": negative_probs.tolist()
            if negative_probs is not None
            else np.ones_like(negative_pids, dtype=np.float32).tolist(),
        }

        if self.return_missing:
            sampling_data.update({"missing_pos": missing_pos})
            sampling_data.update({"missing_neg": missing_neg})

        return sampling_data

    def sampling_data_random(self, labels: list[int], idx: int | None = None, *args, **kwargs) -> dict[str, np.ndarray]:
        # pids that have the same labels
        positive_pids = np.unique(np.concatenate([self.pids_for_label[l] for l in labels]))
        positive_pids = positive_pids[positive_pids != idx]
        if not len(positive_pids):
            positive_pids = np.array([idx])
            logger.warning(f"No positive pid with labels {labels} existent.")
        # pids that have different labels
        negative_pids = np.unique(
            np.concatenate([self.pids_for_label[l] for l in self.unique_labels if l not in labels])
        )

        return {
            "positive_pids": positive_pids.tolist(),
            "negative_pids": negative_pids.tolist(),
            "positive_probs": np.ones_like(positive_pids, dtype=np.float32).tolist(),
            "negative_probs": np.ones_like(negative_pids, dtype=np.float32).tolist(),
        }

    def sampling_data_cw_random(self, label: int, idx: int | None = None, *args, **kwargs) -> dict[str, np.ndarray]:
        raise NotImplementedError()


class TripleSamplerDataset:
    """Samples triples online from sampling data."""

    triple_columns = ["positive_pids", "negative_pids", "positive_probs", "negative_probs"]

    def __init__(
        self,
        triples_sampler_cls: type[TripleSamplingDataBuilder | TripleSamplingDataBuilderMLC],
        config: SamplingConfig,
        text_column: str = "text",
        label_column: str = "label",
    ) -> None:
        self.triples_sampler_cls = triples_sampler_cls
        self.config = config
        self.text_column = text_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.sampling_data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        triple = sample_triple(**self.sampling_data[idx], idx=idx, nway=self.config.nway)
        features = self.dataset_processed[triple]
        # unfold features
        return [{k: v[i] for k, v in features.items()} for i in range(len(triple))]

    def create_sampling_data(
        self,
        dataset: Dataset,
        searcher: Searcher,
        tokenizer: PreTrainedTokenizerBase,
        index_path: str,
    ) -> None:
        if self.config.sampling_method == "searched":
            dataset = search_dataset(
                dataset,
                searcher,
                index_path,
                self.config.searcher_sampling_k,
                self.text_column,
                report_stats=True,
            )
            dataset.save_to_disk(os.path.join(index_path, "index_dataset"))

        dataset_with_sampling_pids = self.build_sampling_data(dataset)
        dataset_with_sampling_pids = self.tokenize(dataset_with_sampling_pids, tokenizer)

        self.sampling_data = dataset_with_sampling_pids.select_columns(self.triple_columns)
        self.sampling_data.set_format("pt")
        self.dataset_processed = dataset_with_sampling_pids.remove_columns(self.triple_columns)
        self.dataset_processed.set_format("pt")

    def build_sampling_data(
        self,
        dataset: Dataset,
    ):
        logger.info("build sampling data")

        dataset.set_format(None)
        sampling_data_builder = self.triples_sampler_cls(
            dataset[self.label_column],
            sampling_method=self.config.sampling_method,
            probability_type=self.config.probability_type,
            nway=self.config.nway,
            n_positives=self.config.n_positives,
            n_negatives=self.config.n_negatives,
            return_missing=True,
            positive_always_random=self.config.positive_always_random,
            lower_ranked_positives=self.config.lower_ranked_positives,
        )

        dataset = dataset.map(
            sampling_data_builder,
            input_columns=sampling_data_builder.input_columns,
            with_indices=True,
            remove_columns=sampling_data_builder.input_columns,
            num_proc=32,
            desc="Sampling",
            load_from_cache_file=False,
        )

        if sampling_data_builder.return_missing:
            missing_pos = sum(dataset["missing_pos"]) / len(dataset)
            missing_neg = sum(dataset["missing_neg"]) / len(dataset)
            dataset = dataset.remove_columns(["missing_pos", "missing_neg"])
            if missing_pos:
                logger.warning(f"Missing {missing_pos:.3f} positive matches in sampling.")
            if missing_neg:
                logger.warning(f"Missing {missing_neg:.3f} negative matches in sampling.")

        return dataset

    def tokenize(self, dataset: Dataset, tokenizer) -> Dataset:
        logger.info("tokenize")

        return dataset.map(
            lambda batch: tokenizer(batch, truncation=True),
            input_columns=self.text_column,
            remove_columns=self.text_column,
            batched=True,
            load_from_cache_file=False,
            desc="Tokenize",
        )


@dataclass
class DataCollatorForTriples(DataCollatorWithPadding):
    def __call__(self, features: list[list[dict[str, torch.Tensor]]]) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        nway = len(features[0])

        # concatenate features
        flattened_features = list(itertools.chain(*features))
        # default collate with padding
        batch = super().__call__(flattened_features)
        # unflatten batch
        batch = {k: v.view(batch_size, nway, *v.shape[1:]) for k, v in batch.items()}
        # label = 0, first passage is the positive one
        batch["labels"] = torch.zeros((batch_size,), dtype=torch.int64)

        return batch


def subsample(dataset: Dataset, size: int | float | None, stratify: bool = True, label_column: str = "label"):
    if size is not None:
        logger.info(f"subsample fac={size} from {len(dataset)}")
        return dataset.train_test_split(
            train_size=size,
            stratify_by_column=label_column if stratify else None,
            load_from_cache_file=False,
        )["train"]

    else:
        return dataset


def sample_triple(
    positive_pids: torch.LongTensor,
    negative_pids: torch.LongTensor,
    positive_probs: torch.Tensor,
    negative_probs: torch.Tensor,
    idx: int,
    nway: int,
) -> torch.LongTensor:
    positive_indices = positive_probs.multinomial(1)
    negative_indices = negative_probs.multinomial(nway - 2)

    sampled_positive = positive_pids[positive_indices]
    sampled_negatives = negative_pids[negative_indices]
    idx = torch.LongTensor([idx])

    return torch.cat([idx, sampled_positive, sampled_negatives])


def normalize_probs(probs: np.ndarray, scaler) -> np.ndarray:
    probs = scaler.fit_transform(probs.reshape(-1, 1)).reshape(-1)
    if probs.sum() == 0:
        return None
    else:
        return probs
