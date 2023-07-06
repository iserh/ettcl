if __name__ == "__main__":
    import torch
    import wandb
    from datasets import load_dataset

    from ettcl.encoding import ColBERTEncoder
    from ettcl.modeling import ColBERTModel, ColBERTTokenizer
    from ettcl.searching.colbert_searcher import ColBERTSearcher, ColBERTSearcherConfig
    from ettcl.utils import catchtime
    from ettcl.utils.multiprocessing import run_multiprocessed

    model_path = "./training/imdb/bert-base-uncased/2023-06-30T09:30:28.027860/checkpoint-7500"
    index_path = "./training/imdb/bert-base-uncased/2023-06-30T09:30:28.027860/checkpoint-7500/index"

    dataset = load_dataset("imdb", split="train")

    model = ColBERTModel.from_pretrained(model_path)
    tokenizer = ColBERTTokenizer.from_pretrained(model_path)
    encoder = ColBERTEncoder(model, tokenizer)

    def search(dataset, config, num_proc, k):
        searcher = ColBERTSearcher(index_path, encoder, config)
        dataset = dataset.map(
            run_multiprocessed(searcher.search),
            input_columns="text",
            fn_kwargs={"k": k},
            batched=True,
            batch_size=1_000,
            with_rank=True,
            num_proc=num_proc,
        )

    def profile(dataset, config, num_proc, k):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        idle_memory = torch.cuda.memory_allocated()

        run = wandb.init(
            project="performance-analysis",
            config={"k": k, "num_proc": num_proc, "idle_memory": idle_memory, **config.__dict__},
            save_code=True,
        )

        with catchtime() as time:
            search(dataset, config, num_proc, k)

        run.log({"execution_time": time.time})
        run.finish()

    config = ColBERTSearcherConfig(plaid_stage_2_3_cpu=True)
    profile(dataset, config, num_proc=4, k=256)

    config = ColBERTSearcherConfig(plaid_stage_2_3_cpu=True)
    profile(dataset, config, num_proc=2, k=256)

    config = ColBERTSearcherConfig(plaid_stage_2_3_cpu=False)
    profile(dataset, config, num_proc=4, k=256)

    config = ColBERTSearcherConfig(plaid_stage_2_3_cpu=False)
    profile(dataset, config, num_proc=2, k=256)
