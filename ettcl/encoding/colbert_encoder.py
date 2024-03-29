from __future__ import annotations

from os import PathLike

import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding

from ettcl.encoding.encoder import MultiVectorEncoder, TEncoder
from ettcl.logging.tqdm import tqdm
from ettcl.modeling.modeling_colbert import ColBERTModel
from ettcl.modeling.tokenization_colbert import ColBERTTokenizer


def sort_by_length(dataset: Dataset, lengths: list[int]) -> tuple[Dataset, torch.LongTensor]:
    lengths = torch.LongTensor(lengths)

    indices = lengths.sort().indices
    reverse_indices = indices.sort().indices

    return dataset.select(indices), reverse_indices


def filter_input_ids(features: dict, filter_ids: torch.Tensor):
    filter_mask = ~torch.isin(features["input_ids"], filter_ids)
    return {"length": filter_mask.sum(), **{k: v[~filter_mask] for k, v in features.items()}}


class ColBERTEncoder(MultiVectorEncoder):
    def __init__(self, model: ColBERTModel, tokenizer: ColBERTTokenizer, skiplist: list[int] | None = None) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.use_gpu = False

        self.skiplist = torch.tensor(skiplist) if skiplist is not None else None
        self.data_collator = DataCollatorWithPadding(tokenizer)

    @property
    def embedding_dim(self) -> int:
        return self.model.output_dimensionality

    def cuda(self: TEncoder, device: int | None = None) -> TEncoder:
        self.model.cuda(device)
        self.use_gpu = True
        return self

    def cpu(self: TEncoder) -> TEncoder:
        self.model.cpu()
        self.use_gpu = False
        return self

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike | None,
        *,
        model_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
    ) -> TEncoder:
        model = ColBERTModel.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
        tokenizer = ColBERTTokenizer.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)
        return cls(model, tokenizer)

    def encode_passages(
        self,
        passages: list[str],
        *,
        batch_size: int = 256,
        to_cpu: bool = True,
        progress_bar: bool = True,
        **unused_kwargs,
    ) -> tuple[torch.FloatTensor, list[int]]:
        assert len(passages) > 0, "No passages provided"
        self.model.eval()

        encodings = self.tokenizer(
            passages,
            mode="doc",
            truncation="longest_first",
            return_length=True,
        )

        lengths = encodings.pop("length")
        dataset = Dataset.from_dict(encodings)
        dataset.set_format("torch")

        if self.skiplist is not None:
            dataset = dataset.map(lambda features: filter_input_ids(features, self.skiplist))
            lengths = dataset["length"]
            dataset = dataset.remove_columns("length")

        dataset, reverse_indices = sort_by_length(dataset, lengths)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=self.data_collator)

        with torch.inference_mode():
            embeddings, masks = [], []
            for input_dict in tqdm(dataloader, desc="Encoding", disable=not progress_bar):
                if self.use_gpu:
                    input_dict = {k: t.cuda() for k, t in input_dict.items()}

                D, mask = self.model(**input_dict)[:2]

                if D.is_cuda:
                    D = D.half()
                if to_cpu:
                    D, mask = D.cpu(), mask.cpu()

                embeddings.extend(D)
                masks.extend(mask.bool())

            embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)[reverse_indices]
            masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)[reverse_indices]

            doc_lengths = masks.sum(-1)
            embeddings_flattened = embeddings.view(-1, embeddings.size(-1))[masks.flatten()]

            assert embeddings_flattened.size(0) == doc_lengths.sum()

        return embeddings_flattened, doc_lengths.tolist()

    def encode_queries(
        self,
        queries: list[str],
        *,
        batch_size: int = 32,
        to_cpu: bool = False,
        keepdims: str = "no_pad",
        progress_bar: bool = True,
        **unused_kwargs,
    ) -> torch.FloatTensor:
        assert len(queries) > 0, "No queries provided"
        self.model.eval()

        encodings = self.tokenizer(
            queries,
            mode="query",
            truncation="longest_first",
            return_length=True,
        )

        lengths = encodings.pop("length")
        dataset = Dataset.from_dict(encodings)
        dataset.set_format("torch")

        if self.skiplist is not None:
            dataset = dataset.map(lambda features: filter_input_ids(features, self.skiplist))
            lengths = dataset["length"]
            dataset = dataset.remove_columns("length")

        dataset, reverse_indices = sort_by_length(dataset, lengths)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=self.data_collator)

        with torch.inference_mode():
            embeddings = []
            for input_dict in tqdm(dataloader, desc="Encoding", disable=not progress_bar):
                if self.use_gpu:
                    input_dict = {k: t.cuda() for k, t in input_dict.items()}

                Q = self.model(**input_dict)[0]

                if to_cpu:
                    Q = Q.cpu()

                embeddings.extend(Q)

            if keepdims == "no_pad":
                return [embeddings[r_idx] for r_idx in reverse_indices]
            else:
                raise NotImplementedError()
