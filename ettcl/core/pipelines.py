from typing import Literal
from ettcl.modeling import ColBERTModel, ColBERTTokenizer, SentenceColBERTModel, SentenceTokenizer, ColBERTConfig, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoTokenizer
from ettcl.indexing import FaissSingleVectorIndexer, ColBERTIndexer, ColBERTIndexerConfig, FaissIndexerConfig
from ettcl.searching import FaissSingleVectorSearcher, ColBERTSearcher, ColBERTSearcherConfig
from ettcl.encoding import ColBERTEncoder, STEncoder
from ettcl.utils import split_into_sentences, chunked, knn_classify
from transformers.modeling_outputs import SequenceClassifierOutput
from logging import getLogger
import os
import torch
from ettcl.core.search import search_dataset
from datasets import Dataset


class ClassificationPipeline:
    text_column: Literal['text'] = 'text'
    label_column: Literal['label'] = 'label'

    def __init__(
        self,
        model_name_or_path: str,
        model_config_kwargs: dict = {},
        architecture: Literal['colbert', 'scolbert', 'sbert', 'bert'] = 'bert',
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
            case 'bert':
                self.model_config = AutoConfig.from_pretrained(model_name_or_path, **model_config_kwargs)
                self.model = BertForSequenceClassification.from_pretrained(model_name_or_path, config=self.model_config)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
            case 'sbert':
                self.model = SentenceTransformer(model_name_or_path)
                self.model._first_module().tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
                self.tokenizer = self.model._first_module()
                self.encoder = STEncoder(self.model, **encoder_kwargs)
            case 'colbert':
                self.model_config = ColBERTConfig.from_pretrained(model_name_or_path, **model_config_kwargs)
                self.model = ColBERTModel.from_pretrained(model_name_or_path, config=self.model_config)
                self.tokenizer = ColBERTTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
                self.encoder = ColBERTEncoder(self.model, self.tokenizer, **encoder_kwargs)
            case 'scolbert':
                self.model_config = ColBERTConfig.from_pretrained(model_name_or_path, **model_config_kwargs)
                self.model = SentenceColBERTModel.from_pretrained(model_name_or_path, config=self.model_config)
                self.tokenizer = SentenceTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
                self.encoder = ColBERTEncoder(self.model, self.tokenizer, **encoder_kwargs)
                # only needed for sentence colbert
                self.num_sentences = kwargs.pop('num_sentences', 16)

        # create indexer and searcher
        match self.architecture:
            case 'sbert':
                self.indexer_config = FaissIndexerConfig(**indexer_kwargs)
                self.indexer = FaissSingleVectorIndexer(self.encoder, self.indexer_config)
                self.searcher = FaissSingleVectorSearcher(None, self.encoder)
                self.k = kwargs.pop('k', 10)
            case 'colbert' | 'scolbert':
                self.indexer_config = ColBERTIndexerConfig(**indexer_kwargs)
                self.indexer = ColBERTIndexer(self.encoder, self.indexer_config)
                self.searcher_config = ColBERTSearcherConfig(**searcher_kwargs)
                self.searcher = ColBERTSearcher(None, self.encoder, self.searcher_config)
                self.k = kwargs.pop('k', 10)

    def train_index(self, dataset: Dataset, text_column: str = 'text', label_column: str = 'label') -> None:
        assert text_column in dataset.column_names, f"Column {text_column} not in dataset columns (found {dataset.column_names})"
        assert label_column in dataset.column_names, f"Column {label_column} not in dataset columns (found {dataset.column_names})"

        self.index_dataset = dataset.select_columns([text_column, label_column])
        self.index_dataset = self.index_dataset.rename_columns({text_column: self.text_column, label_column: self.label_column})

        if self.architecture == 'scolbert':
            self.index_dataset = self.index_dataset.map(
                lambda text: {'text': split_into_sentences(text, self.num_sentences)}, input_columns='text'
            )

        self.index_path = os.path.join(self.o_dir, f"index_{self.architecture}")
        torch.cuda.empty_cache()
        self.indexer.index(self.index_path, self.index_dataset[self.text_column])

    def get_embeddings(self, passages: list[str], mode: str = 'doc') -> list[torch.Tensor] | torch.Tensor:
        match self.architecture:
            case 'sbert':
                return self.encoder.encode_passages(passages, to_cpu=True)['embeddings']

            case 'colbert' | 'scolbert':
                if self.architecture == 'scolbert':
                    passages = list(split_into_sentences(passages, self.num_sentences))

                embs, lengths = self.encoder.encode_passages(passages, to_cpu=True)
                offsets = torch.tensor([0, *lengths])

                return [embs[offset:endpos] for offset, endpos in zip(offsets[:-1], offsets[1:])]

    def predict(self, passages: list[str], use_cached_result: bool = False) -> torch.LongTensor:
        match self.architecture:
            case 'bert':
                return self._head_predict(passages)
            case 'sbert' | 'colbert':
                return self._knn_predict(passages, use_cached_result)
            case 'scolbert':
                passages = list(split_into_sentences(passages, self.num_sentences))
                return self._knn_predict(passages, use_cached_result)

    def _head_predict(self, passages: list[str]) -> torch.LongTensor:

        self.model.cuda()
        self.model.eval()
        preds = []
        for passage_batch in chunked(passages, 256):
            encodings = self.tokenizer(passage_batch, truncation=True, padding=True, return_tensors='pt')
            out: SequenceClassifierOutput = self.model(**encodings.to(self.model.device))
            preds.append(out.logits.argmax(-1).cpu())

        self.model.cpu()

        return torch.cat(preds)

    def _knn_predict(self, passages: list[str], use_cached_result: bool = True) -> torch.LongTensor:

        if not hasattr(self, 'index_path'):
            raise RuntimeError("Index needs to be trained first.")

        if not hasattr(self, 'search_result') or not use_cached_result:
            self.search_result = Dataset.from_dict({'text': passages})
            torch.cuda.empty_cache()
            self.search_result = search_dataset(self.search_result, self.searcher, self.index_path, k=self.k)

        return knn_classify(self.index_dataset, self.search_result, k=self.k, label_column=self.label_column)
