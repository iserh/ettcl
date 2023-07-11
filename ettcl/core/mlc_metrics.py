import torch
from torch.nn.utils.rnn import pad_sequence


def multi_hot(t: list[torch.LongTensor], num_labels: int) -> torch.BoolTensor:
    batch_size = len(t)

    t_padded = pad_sequence(t, True, -1)

    multihot = torch.zeros(batch_size, num_labels + 1, dtype=torch.bool)
    multihot[torch.arange(batch_size).unsqueeze(1), t_padded] = True
    multihot = multihot[:, :-1]

    return multihot


class MLCMetrics:
    def __init__(self, num_labels: int, compute_per_class_metrics: bool = False) -> None:
        self.compute_per_class_metrics = compute_per_class_metrics
        self.num_labels = num_labels
        self.num_examples = 0
        self.acc = torch.tensor(0)
        self.ham = torch.tensor(0)
        self.tp = torch.zeros(num_labels)
        self.fp = torch.zeros(num_labels)
        self.fn = torch.zeros(num_labels)
        self.tn = torch.zeros(num_labels)

    def update(self, eval_pred: list[torch.LongTensor]) -> torch.Tensor:
        preds, labels = eval_pred
        # convert to multi-hot
        preds = multi_hot(preds, self.num_labels)
        labels = multi_hot(labels, self.num_labels)

        self.num_examples += len(preds)
        # exact match / subset accuracy
        self.acc += (preds == labels).all(-1).sum()
        # hamming loss
        self.ham += (preds ^ labels).sum()
        # confusion matrix
        self.tp += (preds & labels).sum(axis=0)
        self.fp += (preds & ~labels).sum(axis=0)
        self.fn += (~preds & labels).sum(axis=0)
        self.tn += (~preds & ~labels).sum(axis=0)

    def compute(self):
        # build global confusion matrix
        total_fp = self.fp.sum()
        total_fn = self.fn.sum()
        total_tp = self.tp.sum()
        total_tn = self.tn.sum()

        # compute metrics per class
        p = self.tp / (self.tp + self.fp + 1e-5)
        r = self.tp / (self.tp + self.fn + 1e-5)
        f = 2.0 * (p * r) / (p + r + 1e-5)

        metrics = {}
        if self.compute_per_class_metrics:
            # build per-class metrics dict
            for i, label in self.h_config.id2label.items():
                metrics[f"{label}/precision"] = p[i].item()
                metrics[f"{label}/recall"] = r[i].item()
                metrics[f"{label}/f1"] = f[i].item()
                metrics[f"{label}/confusion/tp"] = self.tp[i].item()
                metrics[f"{label}/confusion/fp"] = self.fp[i].item()
                metrics[f"{label}/confusion/tn"] = self.tn[i].item()
                metrics[f"{label}/confusion/fn"] = self.fn[i].item()

        # add global confusion matrix
        metrics["confusion/tp"] = total_tp.item()
        metrics["confusion/fp"] = total_fp.item()
        metrics["confusion/tn"] = total_tn.item()
        metrics["confusion/fn"] = total_fn.item()

        # compute average accuracy
        metrics["accuracy"] = (self.acc / self.num_examples).item()
        metrics["hamming"] = (self.ham / (self.num_examples * self.num_labels)).item()

        # compute average metrics
        # micro average precision and recall
        avg_p = total_tp / (total_tp + total_fp + 1e-5)
        avg_r = total_tp / (total_tp + total_fn + 1e-5)

        metrics = metrics | {
            "precision/micro": avg_p.item(),
            "precision/macro": p.mean().item(),
            "recall/micro": avg_r.item(),
            "recall/macro": r.mean().item(),
            "f1/micro": (2.0 * (avg_p * avg_r) / (avg_p + avg_r + 1e-5)).item(),
            "f1/macro": f.mean().item(),
        }

        return metrics
