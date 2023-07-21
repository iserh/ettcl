import torch
from sentence_transformers import SentenceTransformer

from ettcl.encoding.encoder import SingleVectorEncoder, TEncoder
from ettcl.utils.utils import chunked


class STEncoder(SingleVectorEncoder):
    def __init__(self, model: SentenceTransformer, normalize_embeddings: bool = True) -> None:
        super().__init__()
        self.model = model
        self.device = "cpu"
        self.normalize = normalize_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.model.modules[0].config.hidden_size

    def cuda(self: TEncoder, device: int | None = None) -> TEncoder:
        self.model.cuda(device)
        self.device = self.model.device
        return self

    def cpu(self: TEncoder) -> TEncoder:
        self.model.cpu()
        self.device = self.model.device
        return self

    def encode_passages(
        self,
        passages: list[str],
        *,
        batch_size: int = 256,
        to_cpu: bool = True,
        progress_bar: bool = True,
        return_dict: bool = False,
        **unused_kwargs,
    ) -> torch.FloatTensor:
        assert len(passages) > 0, "No passages provided"
        self.model.eval()

        embeddings = []
        for megabatch in chunked(passages, n=batch_size * 4):
            batch_embeddings = self.model.encode(
                megabatch,
                batch_size=batch_size,
                show_progress_bar=progress_bar,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=self.normalize,
            )

            if to_cpu:
                batch_embeddings = batch_embeddings.cpu()

            embeddings.append(batch_embeddings)

        embeddings = torch.cat(embeddings)

        if not return_dict:
            return embeddings

        return {"embeddings": embeddings}

    def encode_queries(
        self,
        queries: list[str],
        *,
        batch_size: int = 32,
        to_cpu: bool = False,
        progress_bar: bool = True,
        return_dict: bool = False,
        **unused_kwargs,
    ) -> torch.FloatTensor:
        assert len(queries) > 0, "No queries provided"
        return self.encode_passages(
            passages=queries,
            batch_size=batch_size,
            to_cpu=to_cpu,
            progress_bar=progress_bar,
            return_dict=return_dict,
        )
