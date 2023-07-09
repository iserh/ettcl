import itertools
import random
import re
from collections.abc import Iterable, Iterator
from logging import getLogger
from time import perf_counter

import numpy as np
import torch

logger = getLogger(__name__)
Devices = int | bool | list[int] | list[str] | None


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


def split_into_sentences(text: str, max_num_sentences: int) -> list[str]:
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
