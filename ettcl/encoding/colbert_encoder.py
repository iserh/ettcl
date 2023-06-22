import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding

from ettcl.encoding.encoder import Encoder
from ettcl.logging.tqdm import tqdm
from ettcl.modeling.modeling_colbert import ColBERTModel
from ettcl.modeling.tokenization_colbert import ColBERTTokenizer
from ettcl.logging.tqdm import tqdm


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

        self.doc_collator = DataCollatorWithPadding(
            tokenizer, padding="longest", max_length=tokenizer.doc_maxlen
        )
        self.query_collator = DataCollatorWithPadding(
            tokenizer, padding="longest", max_length=tokenizer.query_maxlen
        )

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
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=self.doc_collator
        )

        with torch.inference_mode():

            all_embs, all_doclens = [], []
            for input_dict in tqdm(dataloader, desc="Encoding", disable=not progress_bar):
                if self.use_gpu:
                    input_dict = {k: t.cuda() for k, t in input_dict.items()}

                mask = input_dict["attention_mask"]

                D = self.model(**input_dict)[0]

                if D.is_cuda:
                    D = D.half()

                D = D.view(-1, D.shape[-1])[mask.bool().flatten()]
                doclens = mask.sum(-1)

                if to_cpu:
                    D, doclens = D.cpu(), doclens.cpu()

                all_embs.append(D)
                all_doclens.append(doclens)

            all_embs = torch.cat(all_embs)
            all_doclens = torch.cat(all_doclens).tolist()

        return all_embs, all_doclens

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        to_cpu: bool = False,
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
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=self.query_collator
        )

        with torch.inference_mode():
            all_embs = []

            for input_dict in tqdm(dataloader, desc="Encoding", disable=not progress_bar):
                if self.use_gpu:
                    input_dict = {k: t.cuda() for k, t in input_dict.items()}

                Q = self.model(**input_dict)[0]

                if to_cpu:
                    Q = Q.cpu()

                all_embs.extend(Q)

            all_embs = torch.nn.utils.rnn.pad_sequence(all_embs, batch_first=True)

        return all_embs[reverse_indices]
