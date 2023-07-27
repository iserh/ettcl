import os
from logging import getLogger
from typing import Literal

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from ettcl.core.search import search_dataset
from ettcl.core.vis import explain_scores
from ettcl.encoding import ColBERTEncoder, STEncoder
from ettcl.indexing import (
    ColBERTIndexer,
    ColBERTIndexerConfig,
    FaissIndexerConfig,
    FaissSingleVectorIndexer,
)
from ettcl.modeling import (
    BertForSequenceClassification,
    ColBERTConfig,
    ColBERTModel,
    ColBERTTokenizer,
    SentenceColBERTModel,
    SentenceTokenizer,
)
from ettcl.searching import ColBERTSearcher, ColBERTSearcherConfig, FaissSingleVectorSearcher
from ettcl.utils import chunked, knn_classify, split_into_sentences


class ClassificationPipeline:
    text_column: Literal["text"] = "text"
    label_column: Literal["label"] = "label"

    def __init__(
        self,
        model_name_or_path: str,
        model_config_kwargs: dict = {},
        architecture: Literal["colbert", "scolbert", "sbert", "bert"] = "bert",
        encoder_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
        indexer_kwargs: dict = {},
        searcher_kwargs: dict = {},
        output_directory: str | os.PathLike = "outputs",
        **kwargs,
    ) -> None:
        self.logger = getLogger(self.__class__.__name__)
        self.architecture = architecture
        self.o_dir = output_directory

        # load model, tokenizer and create encoder
        match self.architecture:
            case "bert":
                self.model_config = AutoConfig.from_pretrained(model_name_or_path, **model_config_kwargs)
                self.model = BertForSequenceClassification.from_pretrained(model_name_or_path, config=self.model_config)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
            case "sbert":
                self.model = SentenceTransformer(model_name_or_path)
                self.model._first_module().tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, **tokenizer_kwargs
                )
                self.tokenizer = self.model._first_module()
                self.encoder = STEncoder(self.model, **encoder_kwargs)
            case "colbert":
                self.model_config = ColBERTConfig.from_pretrained(model_name_or_path, **model_config_kwargs)
                self.model = ColBERTModel.from_pretrained(model_name_or_path, config=self.model_config)
                self.tokenizer = ColBERTTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
                self.encoder = ColBERTEncoder(self.model, self.tokenizer, **encoder_kwargs)
            case "scolbert":
                self.model_config = ColBERTConfig.from_pretrained(model_name_or_path, **model_config_kwargs)
                self.model = SentenceColBERTModel.from_pretrained(model_name_or_path, config=self.model_config)
                self.tokenizer = SentenceTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
                self.encoder = ColBERTEncoder(self.model, self.tokenizer, **encoder_kwargs)
                # only needed for sentence colbert
                self.num_sentences = kwargs.pop("num_sentences", 16)

        # create indexer and searcher
        match self.architecture:
            case "sbert":
                self.indexer_config = FaissIndexerConfig(**indexer_kwargs)
                self.indexer = FaissSingleVectorIndexer(self.encoder, self.indexer_config)
                self.searcher = FaissSingleVectorSearcher(None, self.encoder)
                self.k = kwargs.pop("k", 10)
            case "colbert" | "scolbert":
                self.indexer_config = ColBERTIndexerConfig(**indexer_kwargs)
                self.indexer = ColBERTIndexer(self.encoder, self.indexer_config)
                self.searcher_config = ColBERTSearcherConfig(**searcher_kwargs)
                self.searcher = ColBERTSearcher(None, self.encoder, self.searcher_config)
                self.k = kwargs.pop("k", 10)

    def train_index(
        self, dataset: Dataset, index_path: str | None = None, text_column: str = "text", label_column: str = "label"
    ) -> None:
        assert (
            text_column in dataset.column_names
        ), f"Column {text_column} not in dataset columns (found {dataset.column_names})"
        assert (
            label_column in dataset.column_names
        ), f"Column {label_column} not in dataset columns (found {dataset.column_names})"

        self.index_dataset = dataset.select_columns([text_column, label_column])
        self.index_dataset = self.index_dataset.rename_columns(
            {text_column: self.text_column, label_column: self.label_column}
        )

        self.label_names = self.index_dataset.features[label_column].names

        if self.architecture == "scolbert":
            self.index_dataset = self.index_dataset.map(
                lambda text: {"text": split_into_sentences(text, self.num_sentences)}, input_columns="text"
            )

        if index_path is not None:
            self.index_path = index_path
        else:
            self.index_path = os.path.join(self.o_dir, f"index_{self.architecture}")
            torch.cuda.empty_cache()
            self.indexer.index(self.index_path, self.index_dataset[self.text_column])

    def get_embeddings(self, passages: list[str], mode: str = "doc") -> list[torch.Tensor] | torch.Tensor:
        match self.architecture:
            case "sbert":
                return self.encoder.encode_passages(passages, to_cpu=True)["embeddings"]

            case "colbert" | "scolbert":
                if self.architecture == "scolbert":
                    passages = list(split_into_sentences(passages, self.num_sentences))

                embs, lengths = self.encoder.encode_passages(passages, to_cpu=True)
                offsets = torch.tensor([0, *lengths])

                return [embs[offset:endpos] for offset, endpos in zip(offsets[:-1], offsets[1:])]

    def predict(
        self, passages: list[str], labels: list[int] | None = None, use_cached_result: bool = False
    ) -> torch.LongTensor:
        match self.architecture:
            case "bert":
                self.predictions = self._head_predict(passages, labels)
            case "sbert" | "colbert":
                self.predictions = self._knn_predict(passages, labels, use_cached_result)
            case "scolbert":
                passages = list(split_into_sentences(passages, self.num_sentences))
                self.predictions = self._knn_predict(passages, labels, use_cached_result)

        return self.predictions

    def explain_prediction(self, idx: int, match_idx: int, true_labels: list[int] | None = None):
        if not hasattr(self, "search_result"):
            raise RuntimeError(
                "This pipeline has no search results stored. This can either be because the selected "
                f"architecture '{self.architecture}' does not support explanations or you forgot to call "
                "'predict' first."
            )

        test_example = self.search_result[idx]
        matches = self.index_dataset[self.search_result[idx]["match_pids"]]

        if true_labels is not None:
            true_label = self.label_names[true_labels[idx]]
        elif self.label_column in test_example:
            true_label = self.label_names[test_example[self.label_column]]
        else:
            true_label = "?"

        pred_label = self.label_names[matches[self.label_column][match_idx]]

        print(f"Query {idx}: {test_example['text']}")
        print(f"True Label: {true_label}, predicted label: {pred_label}")
        print(f"Match {match_idx}: {matches[self.text_column][match_idx]}")

        return explain_scores(
            model=self.model,
            tokenizer=self.tokenizer,
            query=test_example[self.text_column],
            doc=matches[self.text_column][match_idx],
        )

    def _head_predict(self, passages: list[str], labels: list[int] | None = None) -> torch.LongTensor:
        self.model.cuda()
        self.model.eval()
        preds = []
        for passage_batch in chunked(passages, 256):
            encodings = self.tokenizer(passage_batch, truncation=True, padding=True, return_tensors="pt")
            out: SequenceClassifierOutput = self.model(**encodings.to(self.model.device))
            preds.append(out.logits.argmax(-1).cpu())

        self.model.cpu()

        return torch.cat(preds)

    def _knn_predict(
        self, passages: list[str], labels: list[int] | None = None, use_cached_result: bool = True
    ) -> torch.LongTensor:
        if not hasattr(self, "index_path"):
            raise RuntimeError("Index needs to be trained first.")

        if not hasattr(self, "search_result") or not use_cached_result:
            self.search_result = Dataset.from_dict({"text": passages})
            torch.cuda.empty_cache()
            self.search_result = search_dataset(self.search_result, self.searcher, self.index_path, k=self.k)

        if labels is not None:
            labels = labels.tolist() if isinstance(labels, torch.Tensor) else labels
            self.search_result = self.search_result.add_column(self.label_column, labels)

        return knn_classify(self.index_dataset, self.search_result, k=self.k, label_column=self.label_column)
