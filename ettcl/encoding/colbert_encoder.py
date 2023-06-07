import torch
from colbert.infra.config.config import ColBERTConfig
from colbert.infra.run import Run
from colbert.modeling.checkpoint import Checkpoint
from colbert.utils.utils import batch

from ettcl.encoding.base_encoder import BaseEncoder


class ColBERTEncoder(BaseEncoder):
    def __init__(
        self, checkpoint: str, config: ColBERTConfig | None = None, use_gpu: bool = False
    ) -> None:
        super().__init__(use_gpu)
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)
        self.config: ColBERTConfig = ColBERTConfig.from_existing(self.checkpoint_config, config)
        self.config.configure(checkpoint=checkpoint)

        # only way to disable colbert's gpu usage is to limit the visible gpus (Usually the launcher does this by setting CUDA_VISIBLE_DEVICES)
        if not self.use_gpu:
            self.config.configure(total_visible_gpus=0)

        self.checkpoint = Checkpoint(checkpoint, colbert_config=self.config)
        if self.use_gpu:
            self.checkpoint.cuda()

    def encode_passages(self, passages: list[str]) -> tuple[torch.FloatTensor, list[int]]:
        Run().print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            for passages_batch in batch(passages, self.config.bsize * 50):
                embs_, doclens_ = self.checkpoint.docFromText(
                    passages_batch,
                    bsize=self.config.bsize,
                    keep_dims="flatten",
                    showprogress=not self.use_gpu,
                )
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

        return embs, doclens

    def encode_queries(self, queries: list[str]) -> torch.FloatTensor:
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, showprogress=not self.use_gpu)

        return Q
