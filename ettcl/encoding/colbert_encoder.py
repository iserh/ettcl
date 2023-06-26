import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding

from ettcl.encoding.encoder import Encoder
from ettcl.logging.tqdm import tqdm
from ettcl.modeling.modeling_colbert import ColBERTModel
from ettcl.modeling.tokenization_colbert import ColBERTTokenizer


def sort_by_length(dataset: Dataset, lengths: list[int]) -> tuple[Dataset, torch.LongTensor]:
    lengths = torch.LongTensor(lengths)

    indices = lengths.sort().indices
    reverse_indices = indices.sort().indices

    return dataset.select(indices), reverse_indices


class ColBERTEncoder(Encoder):
    def __init__(self, model: ColBERTModel, tokenizer: ColBERTTokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.use_gpu = False

        self.data_collator = DataCollatorWithPadding(tokenizer)

    @property
    def embedding_dim(self) -> int:
        return self.model.output_dimensionality

    def cuda(self, device: int | None = None) -> Encoder:
        self.model.cuda(device)
        self.use_gpu = True
        return self

    def cpu(self) -> Encoder:
        self.model.cpu()
        self.use_gpu = False
        return self

    def encode_passages(
        self,
        passages: list[str],
        batch_size: int = 256,
        to_cpu: bool = True,
        progress_bar: bool = True,
    ) -> tuple[torch.FloatTensor, list[int]]:
        assert len(passages) > 0, "No passages provided"

        encodings = self.tokenizer(
            passages,
            mode="doc",
            add_special_tokens=True,
            truncation="longest_first",
            return_length=True,
        )

        lengths = encodings.pop("length")
        dataset = Dataset.from_dict(encodings)
        dataset, reverse_indices = sort_by_length(dataset, lengths)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=self.data_collator)

        with torch.inference_mode():
            embeddings, masks = [], []
            for input_dict in tqdm(dataloader, desc="Encoding", disable=not progress_bar):
                if self.use_gpu:
                    input_dict = {k: t.cuda() for k, t in input_dict.items()}

                D = self.model(**input_dict)[0]
                mask = input_dict["attention_mask"]

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
        batch_size: int = 32,
        to_cpu: bool = False,
        keepdims: str = "no_pad",
        progress_bar: bool = True,
    ) -> torch.FloatTensor:
        assert len(queries) > 0, "No queries provided"

        encodings = self.tokenizer(
            queries,
            mode="query",
            add_special_tokens=True,
            truncation="longest_first",
            return_length=True,
        )

        lengths = encodings.pop("length")
        dataset = Dataset.from_dict(encodings)
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
