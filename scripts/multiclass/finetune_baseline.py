#!/usr/bin/env python
import os
from datetime import datetime
from pathlib import Path

import wandb
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from ettcl.logging import configure_logger
from ettcl.utils import seed_everything
from ettcl.modeling import BertForSequenceClassification


def compute_metrics(eval_prediction: EvalPrediction) -> dict:
    preds = eval_prediction.predictions.argmax(-1)
    targets = eval_prediction.label_ids

    metrics = {}
    metrics[f"accuracy"] = accuracy_score(y_pred=preds, y_true=targets)
    metrics[f"precision/micro"] = precision_score(y_pred=preds, y_true=targets, average="micro")
    metrics[f"precision/macro"] = precision_score(y_pred=preds, y_true=targets, average="macro")
    metrics[f"recall/micro"] = recall_score(y_pred=preds, y_true=targets, average="micro")
    metrics[f"recall/macro"] = recall_score(y_pred=preds, y_true=targets, average="macro")
    metrics[f"f1/micro"] = f1_score(y_pred=preds, y_true=targets, average="micro")
    metrics[f"f1/macro"] = f1_score(y_pred=preds, y_true=targets, average="macro")

    return metrics


def main(params: dict, log_level: str | int = "INFO") -> None:
    configure_logger(log_level)

    seed = params["seed"]["value"] if "seed" in params.keys() else 12345
    seed_everything(seed)

    config = params["config"]["value"]
    output_dir = (
        Path.cwd()
        / "training"
        / os.path.basename(params["dataset"]["value"])
        / "baseline"
        / os.path.basename(params["model"]["value"])
        / datetime.now().isoformat()
    )

    output_dir.mkdir(exist_ok=True, parents=True)

    dataset = load_from_disk(params["dataset"]["value"])
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.rename_columns({config["text_column"]: "text", config["label_column"]: "label"})
    val_dataset = val_dataset.rename_columns({config["text_column"]: "text", config["label_column"]: "label"})
    test_dataset = test_dataset.rename_columns({config["text_column"]: "text", config["label_column"]: "label"})
    if "remove_columns" in config:
        train_dataset = train_dataset.remove_columns(config["remove_columns"])
        val_dataset = val_dataset.remove_columns(config["remove_columns"])
        test_dataset = test_dataset.remove_columns(config["remove_columns"])

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

    num_labels = len(train_dataset["label"].unique())
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(params["model"]["value"])

    train_dataset = train_dataset.map(lambda batch: tokenizer(batch["text"], truncation=True), batched=True)
    val_dataset = val_dataset.map(lambda batch: tokenizer(batch["text"], truncation=True), batched=True)
    test_dataset = test_dataset.map(lambda batch: tokenizer(batch["text"], truncation=True), batched=True)

    params["training"]["value"].pop("output_dir", None)
    params["training"]["value"].pop("seed", None)
    training_args = TrainingArguments(output_dir=output_dir, seed=seed, **params["training"]["value"])
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    run_config = {
        "seed": seed,
        "dataset": os.path.basename(params["dataset"]["value"]),
        "architecture": "BERT",
        "model": params["model"]["value"],
        "training": training_args.to_dict(),
        "config": config,
    }

    run = wandb.init(
        project=config["project"],
        dir=output_dir,
        config=run_config,
        save_code=True,
    )
    run.log_code(
        ".",
        include_fn=lambda path: path.endswith(".py")
        or path.endswith(".cpp")
        or path.endswith(".cu")
        or path.endswith(".yml"),
    )

    trainer.train()

    best_model_checkpoint = trainer.state.best_model_checkpoint
    if best_model_checkpoint is not None and os.path.exists(best_model_checkpoint):
        os.rename(best_model_checkpoint, os.path.join(training_args.output_dir, "model_best"))

    trainer.evaluate(test_dataset, metric_key_prefix="test")


if __name__ == "__main__":
    from argparse import ArgumentParser

    import yaml

    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the train config yaml file.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    with open(args.path, "r") as f:
        params = yaml.load(f, yaml.SafeLoader)

    main(params, args.log_level)
