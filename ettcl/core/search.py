from logging import getLogger

import torch
from datasets import Dataset

from ettcl.logging import profile_memory
from ettcl.searching import Searcher
from ettcl.utils.multiprocessing import run_multiprocessed

logger = getLogger(__name__)


@profile_memory
def search_dataset(
    dataset: Dataset,
    searcher: Searcher,
    index_path: str,
    k: int,
    text_column: str = "text",
    report_stats: bool = False,
) -> Dataset:
    logger.info("search dataset")

    searcher.index_path = index_path
    dataset.set_format(None)

    dataset = dataset.map(
        run_multiprocessed(searcher.search),
        input_columns=text_column,
        fn_kwargs={"k": k},
        batched=True,
        num_proc=torch.cuda.device_count(),
        with_rank=True,
        desc="Searching",
    )

    if report_stats:
        # log some statistics about how many matches were found
        dataset.set_format("numpy")
        dataset = dataset.map(lambda pids: {"len_": len(pids)}, input_columns="match_pids")
        avg_matches = dataset["len_"].mean()
        dataset = dataset.remove_columns("len_")
        logger.info(f"average #matches: {avg_matches}")

    return dataset
