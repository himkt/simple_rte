import json
import logging
from typing import Dict, List

import torch
import torch.utils.data
import tqdm
from rich.logging import RichHandler
from transformers import (AdamW, BertForSequenceClassification,
                          BertTokenizerFast, EvalPrediction, Trainer,
                          TrainingArguments)
from transformers.tokenization_utils_base import BatchEncoding

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
logger = logging.getLogger("rich")


def read_jsonl(path: str, tokenizer: BertTokenizerFast, with_labels: bool):
    items = {
        "premises": [],
        "hypothesises": [],
        "labels": [] if with_labels else None,
        "label_ids": [] if with_labels else None,
    }

    for line in tqdm.tqdm(open(path)):
        item = json.loads(line)
        items["premises"].append(item["premise"])
        items["hypothesises"].append(item["hypothesis"])
        if with_labels:
            items["labels"].append(item["label"])
            items["label_ids"].append(0 if item["label"] == "entailment" else 1)

    return (
        tokenizer(items["premises"], items["hypothesises"], return_tensors="pt", padding=True, truncation=True),
        items["label_ids"],
    )


class RTEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: BatchEncoding, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def metrics(p: EvalPrediction) -> Dict[str, float]:
    acc = (p.predictions.argmax(axis=1) == p.label_ids).sum() / len(p.label_ids)
    return {"accuracy": acc}


if __name__ == "__main__":
    # tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
    tokenizer = BertTokenizerFast.from_pretrained("textattack/bert-base-uncased-RTE")
    logger.info("Created tokenizer")

    train_texts, train_labels = read_jsonl("./data/train.jsonl", tokenizer, with_labels=True)
    train_dataset = RTEDataset(encodings=train_texts, labels=train_labels)
    logger.info("Loaded dataset for training")

    valid_texts, valid_labels = read_jsonl("./data/val.jsonl", tokenizer, with_labels=True)
    valid_dataset = RTEDataset(encodings=valid_texts, labels=valid_labels)
    logger.info("Loaded dataset for validation")

    # model = BertForSequenceClassification.from_pretrained("bert-large-cased")
    model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-RTE")
    # for param in model.base_model.parameters():
    #     param.requires_grad = False
    # logger.info("Freeze BERT parameters")

    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,
        eps=1e-8,
    )

    config = TrainingArguments(
        output_dir="./results",
        num_train_epochs=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        weight_decay=1e-2,
        logging_dir="./logs",
        evaluation_strategy="epoch",  # [memo] `evaluation_strategy` is needed, not `do_eval`.
    )

    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=metrics,
    )

    trainer.train()
