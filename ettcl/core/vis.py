import torch
from bertviz import head_view

from ettcl.modeling.modeling_colbert import ColBERTModel, colbert_score
from ettcl.modeling.tokenization_colbert import ColBERTTokenizer
from ettcl.modeling.tokenization_sentence_colbert import SentenceTokenizer


def maxsim_view(
    scores: torch.Tensor,
    Q_mask: torch.LongTensor,
    D_mask: torch.LongTensor,
    Q_tokens: list[str] = None,
    D_tokens: list[str] = None,
    max_only: bool = False,
):
    if scores.ndim == 3:
        scores = scores[0]
    if Q_mask.ndim == 2:
        Q_mask = Q_mask[0]
    if D_mask.ndim == 2:
        D_mask = D_mask[0]

    scores[~D_mask.bool()] = -9999
    scores = scores[..., Q_mask.bool()]

    scores_max_only = scores.clone()
    scores_max_only[(scores < scores.max(0, keepdim=True).values)] = -9999

    if max_only:
        scores = scores_max_only.unsqueeze(0)
    else:
        scores = torch.stack([scores_max_only, scores])

    return head_view(
        cross_attention=scores.permute(0, 2, 1).unsqueeze(1).unsqueeze(1),
        encoder_tokens=D_tokens if D_tokens is not None else [f"D[{i}]" for i in range(scores.shape[-1])],
        decoder_tokens=Q_tokens if Q_tokens is not None else [f"Q[{i}]" for i in range(scores.shape[-2])],
        html_action="return",
        layer=0,
    )


def explain_scores(
    model: ColBERTModel,
    tokenizer: ColBERTTokenizer,
    query: str,
    doc: str,
    use_gpu: bool = True,
    max_only: bool = False,
):
    q_inputs = tokenizer(query, mode="query", return_tensors="pt")
    d_inputs = tokenizer(doc, mode="doc", return_tensors="pt")

    if isinstance(tokenizer, SentenceTokenizer):
        q_inputs = {k: torch.stack(v) for k, v in q_inputs.items()}
        d_inputs = {k: torch.stack(v) for k, v in d_inputs.items()}

    query_length = q_inputs["attention_mask"].sum()
    doc_length = d_inputs["attention_mask"].sum()

    q_tokens = tokenizer.tokenize(query, mode="query")[:query_length]
    d_tokens = tokenizer.tokenize(doc, mode="doc")[:doc_length]

    if use_gpu:
        model.cuda()
        q_inputs = {k: v.cuda() for k, v in q_inputs.items()}
        d_inputs = {k: v.cuda() for k, v in d_inputs.items()}

    with torch.inference_mode():
        out = model(**q_inputs)
        Q = out.normalized_output.cpu()
        Q_mask = out.output_mask.cpu()

        out = model(**d_inputs)
        D = out.normalized_output.cpu()
        D_mask = out.output_mask.cpu()

    scores = colbert_score(Q, D)

    return maxsim_view(scores, Q_mask, D_mask, q_tokens, d_tokens, max_only=max_only)
