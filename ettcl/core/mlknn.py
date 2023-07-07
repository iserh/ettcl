import torch

from ettcl.logging import trange


def membership_counting_vector(labels: list[torch.Tensor], num_labels: int):
    mcv = torch.zeros(num_labels, dtype=torch.long)
    indices, counts = torch.cat(labels).unique(return_counts=True)
    mcv[indices] = counts
    return mcv


class MLKNN(object):
    def __init__(self, nn_matrix: torch.LongTensor, y_train: list[torch.LongTensor], k: int = 10, s: float = 1):
        self.nn_matrix = nn_matrix[:, :k]
        self.y_train = y_train
        self.num_labels = torch.cat(y_train).unique().shape[0]
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
                neighbor_labels = [self.y_train[idx] for idx in self.nn_matrix[i]]
                C_xi_l = (torch.cat(neighbor_labels) == l).sum()
                c[C_xi_l, int(l in self.y_train[i])] += 1

            self.posterior[l, :, 1] = (self.s + c[:, 1]) / (self.s * (self.k + 1) + c[:, 1].sum())
            self.posterior[l, :, 0] = (self.s + c[:, 0]) / (self.s * (self.k + 1) + c[:, 0].sum())

    def predict(self, nn_matrix: torch.LongTensor) -> torch.LongTensor:
        nn_matrix = nn_matrix[: self.k]
        neighbor_labels = [self.y_train[idx] for idx in nn_matrix]

        C_t_l = membership_counting_vector(neighbor_labels, self.num_labels)
        y_multihot = (
            self.prior[:, 1] * self.posterior[range(len(C_t_l)), C_t_l, 1]
            > self.prior[:, 0] * self.posterior[range(len(C_t_l)), C_t_l, 0]
        )

        return torch.where(y_multihot)[0]
