import itertools
import json
import os
import random
import re
from collections.abc import Iterable, Iterator
from logging import getLogger
from time import perf_counter
from typing import Generator

import numpy as np
import torch
from datasets import Dataset

logger = getLogger(__name__)
Devices = int | bool | list[int] | list[str] | None


class Checkpoint:
    def __init__(self, path_to_experiment: str) -> None:
        self.experiment = path_to_experiment
        self.index = os.path.join(path_to_experiment, "index_best")

        checkpoints = [name for name in os.listdir(path_to_experiment) if name.startswith("checkpoint")]
        self.latest = os.path.join(path_to_experiment, max((chkpt for chkpt in checkpoints), default=None))

        trainer_state_path = os.path.join(self.latest, "trainer_state.json")
        assert os.path.exists(trainer_state_path), f"{self.latest} is no valid checkpoint."

        with open(trainer_state_path, "r") as fp:
            trainer_state = json.load(fp)

        self.best = trainer_state["best_model_checkpoint"]
        assert os.path.exists(self.best), f"{self.best} does not exist."


def knn_classify(index_dataset_or_labels, test_dataset_or_pids, k: int = 10, label_column: str = "label"):
    if isinstance(index_dataset_or_labels, Dataset):
        index_labels = index_dataset_or_labels.with_format("torch")[label_column]
    else:
        index_labels = torch.tensor(index_dataset_or_labels)

    if isinstance(index_dataset_or_labels, Dataset):
        match_pids = test_dataset_or_pids.with_format("torch")["match_pids"]

    match_pids = torch.nn.utils.rnn.pad_sequence(list(match_pids), batch_first=True, padding_value=-1)
    match_labels = index_labels[match_pids.tolist()]
    match_labels[match_pids == -1] = -1

    knn = match_labels[:, :k]
    y_pred = torch.mode(knn)[0]

    return y_pred


def chunked(iterator: Iterator | Iterable, n: int) -> Iterator[list]:
    if isinstance(iterator, Iterable):
        iterator = iter(iterator)

    while chunk := list(itertools.islice(iterator, n)):
        yield chunk


def seed_everything(seed: int = 12345) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_gpu_list(gpus: Devices) -> list[int]:
    """List of selected gpu devices."""
    # convert to list of selected gpus
    match gpus:
        case bool():
            gpus = list(range(torch.cuda.device_count())) if gpus else []
        case int():
            gpus = list(range(gpus))
        case str():
            gpus = gpus.split(",")
        case None:
            gpus = []

    # cast list items to integer
    gpus = list(map(int, gpus))
    # sort list
    gpus = sorted(list(set(gpus)))

    # make sure only visible devices are selected
    assert all(
        device_idx in range(torch.cuda.device_count()) for device_idx in gpus
    ), f"Not all devices from {gpus} are visible!"

    return gpus


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = (
    "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
)
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r"\.{2,}"


def split_into_sentences(text: str | list[str], max_num_sentences: int) -> list[str] | Generator[list[str], None, None]:
    if isinstance(text, list):
        return map(lambda text: _split_into_sentences(text, max_num_sentences), text)
    else:
        return _split_into_sentences(text, max_num_sentences)


def _split_into_sentences(text: str, max_num_sentences: int) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences[:max_num_sentences]


class catchtime:
    def __init__(self, output_to_console: bool = True, desc: str = "Time"):
        self.output_to_console = output_to_console
        self.desc = desc

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        if self.output_to_console:
            self.readout = f"{self.desc}: {self.time:.3f} seconds"
            logger.info(self.readout)
