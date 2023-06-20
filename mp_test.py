import multiprocessing as mp
import os
import torch
import time

try:
    mp.set_start_method("spawn")
except:
    pass


def worker_fn(nranks, rank):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(nranks)))

    print(f"{rank}> DEVICE_COUNT: {torch.cuda.device_count()}")
    print(f"{rank}> CURRENT_DEVICE: {torch.cuda.current_device()}")

    t = torch.empty(size=(32, 8)).cuda()
    print(f"{rank}> DEVICE: {t.device}")


if __name__ == "__main__":
    print(f"#> DEVICE_COUNT: {torch.cuda.device_count()}")
    print(f"#> CURRENT_DEVICE: {torch.cuda.current_device()}")

    t = torch.empty(size=(32, 8)).cuda()
    print(f"#> DEVICE: {t.device}")

    nranks = 1

    processes = []
    for rank in range(nranks):
        processes.append(mp.Process(target=worker_fn, args=(nranks, rank)))

    for proc in processes:
        print("#> Starting ...")
        proc.start()

    for proc in processes:
        proc.join()
        print("#> Joined")
