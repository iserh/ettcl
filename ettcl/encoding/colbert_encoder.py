import torch
from datasets import Dataset
from tqdm import tqdm

from ettcl.encoding.base_encoder import BaseEncoder
from ettcl.modeling.modeling_colbert import ColBERTModel
from ettcl.modeling.tokenization_colbert import ColBERTTokenizer


class ColBERTEncoder(BaseEncoder):
    def __init__(self, model: ColBERTModel, tokenizer: ColBERTTokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.use_gpu = False

    @property
    def embedding_dim(self) -> int:
        return self.model.output_dimensionality

    def cuda(self) -> BaseEncoder:
        self.model.cuda()
        self.use_gpu = True
        return self

    def cpu(self) -> BaseEncoder:
        self.model.cpu()
        self.use_gpu = False
        return self

    def encode_passages(
        self, passages: list[str], batch_size: int = 128, to_cpu: bool = True
    ) -> tuple[torch.FloatTensor, list[int]]:
        assert len(passages) > 0, "No passages provided"

        # TODO: Could sort by doc length to speed up encoding
        input_data = Dataset.from_dict({"text": passages})
        input_data.set_format("pt")
        input_data = input_data.map(
            lambda batch: self.tokenizer(
                batch["text"],
                mode="doc",
                add_special_tokens=True,
                padding="longest",
                truncation="longest_first",
                return_tensors="pt",
            ),
            batched=True,
            batch_size=1024
            - (
                1024 % batch_size
            ),  # floor to whole of batch_size (important because of padding='longest')
        )
        input_data = input_data.remove_columns("text")

        with torch.inference_mode():
            all_embs, all_doclens = [], []
            for input_dict in tqdm(
                input_data.iter(batch_size), total=len(input_data) // batch_size, desc="Encoding"
            ):
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
        self, queries: list[str], batch_size: int = 128, to_cpu: bool = False
    ) -> torch.FloatTensor:
        assert len(queries) > 0, "No queries provided"

        input_data = Dataset.from_dict({"text": queries})
        input_data.set_format("pt")
        input_data = input_data.map(
            lambda batch: self.tokenizer(
                batch["text"],
                mode="query",
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ),
            batched=True,
            batch_size=1024
            - (
                1024 % batch_size
            ),  # floor to whole of batch_size (important because of padding='longest')
        )
        input_data = input_data.remove_columns("text")

        with torch.inference_mode():
            all_embs = []
            for input_dict in input_data.iter(batch_size):
                if self.use_gpu:
                    input_dict = {k: t.cuda() for k, t in input_dict.items()}

                Q = self.model(**input_dict)[0]

                if to_cpu:
                    Q = Q.cpu()

                all_embs.append(Q)

            all_embs = torch.cat(all_embs)

        return all_embs
