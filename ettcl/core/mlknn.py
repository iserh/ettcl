import torch

from ettcl.logging import trange
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import json

def membership_counting_vector(labels: list[torch.Tensor], num_labels: int):
    mcv = torch.zeros(num_labels, dtype=torch.long)
    indices, counts = labels.unique(return_counts=True)
    mcv[indices] = counts
    return mcv


class MLKNN(object):
    def __init__(self, nn_matrix: list[torch.LongTensor], y_train: list[torch.LongTensor], k: int = 10, s: float = 1):
        self.nn_matrix = pad_sequence(nn_matrix, batch_first=True, padding_value=-1)[:, :k]
        assert self.nn_matrix.shape[1] == k, f"Not enough neighbors, expected {k} but got {self.nn_matrix.shape[1]}"

        self.y_train = pad_sequence(y_train, batch_first=True, padding_value=-1)
        # self.y_train = y_train
        self.num_labels = torch.cat(y_train).max() + 1
        self.num_examples = len(y_train)
        self.k = k
        self.s = s

    def train(self, progress_bar: bool = True):
        self.prior = torch.zeros(self.num_labels, 2)
        self.posterior = torch.zeros(self.num_labels, self.k + 1, 2)

        # compute prior
        cnt = membership_counting_vector(self.y_train, self.num_labels)
        self.prior[:, 1] = (self.s + cnt) / (self.s * 2 + self.num_examples)
        self.prior[:, 0] = 1 - self.prior[:, 1]

        # compute posterior, TODO: avoid for loop?
        for l in trange(self.num_labels, desc="ML-kNN (posterior)", disable=not progress_bar):
            c = torch.zeros(self.k + 1, 2, dtype=torch.long)

            for i in range(len(self.y_train)):
                neighbor_labels = self.y_train[self.nn_matrix[i][self.nn_matrix[i] != -1]]
                C_xi_l = (neighbor_labels == l).sum()
                c[C_xi_l, int(l in self.y_train[i])] += 1

            self.posterior[l, :, 1] = (self.s + c[:, 1]) / (self.s * (self.k + 1) + c[:, 1].sum())
            self.posterior[l, :, 0] = (self.s + c[:, 0]) / (self.s * (self.k + 1) + c[:, 0].sum())

    def predict(self, neighbor_ids: torch.LongTensor) -> torch.LongTensor:
        """Predict single example based on neighbors."""
        neighbor_ids = neighbor_ids[: self.k]
        neighbor_labels = self.y_train[neighbor_ids]
        neighbor_labels = neighbor_labels[neighbor_labels != -1]

        C_t_l = membership_counting_vector(neighbor_labels, self.num_labels)
        y_multihot = (
            self.prior[:, 1] * self.posterior[range(len(C_t_l)), C_t_l, 1]
            > self.prior[:, 0] * self.posterior[range(len(C_t_l)), C_t_l, 0]
        )

        return torch.where(y_multihot)[0]

    def save(self, path: str) -> None:
        path = Path(path)

        torch.save(self.prior, path / "prior.pt")
        torch.save(self.posterior, path / "posterior.pt")

        metadata = {"k": self.k, "s": self.s, "num_labels": self.num_labels, "num_examples": self.num_examples}
        with open(path / "metadata.json", "w") as fp:
            json.dump(metadata, fp)
