import glob
import logging
import os.path
from typing import Any, Callable, Dict
from typing import List

import torch
import torch.utils.data
import tqdm
from rich.logging import RichHandler
from transformers import BertForSequenceClassification
from transformers import BertJapaneseTokenizer
from transformers import EvalPrediction
from transformers import Trainer
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import BatchEncoding
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from optuna import create_study
from optuna import Trial


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
logger = logging.getLogger("rich")


def read_livedoor(path: str) -> Dict[str, Any]:
    label_to_id = {}
    items = []

    for dirname in tqdm.tqdm(glob.glob(os.path.join(path, "*"))):
        if not os.path.isdir(dirname):
            continue

        label = os.path.basename(dirname)
        label_id = label_to_id.get(label, len(label_to_id))
        label_to_id[label] = label_id

        for filename in glob.glob(os.path.join(dirname, "*.txt")):
            if filename == "LICENSE.txt":
                continue

            fp = open(filename)
            fp.readline()  # article_url
            fp.readline()  # timestamp
            text = "".join(line.strip() for line in fp.readlines())
            item = {"label": label, "label_id": label_id, "text": text}
            items.append(item)

    return items


class LivedoorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: BatchEncoding, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def create_objective(model_name: str, num_labels: int) -> Callable:
    def objective(trial: Trial) -> float:
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        for param in model.base_model.parameters():
            param.requires_grad = False
        logger.info("Freeze BERT parameters")

        for key, param in model.named_parameters():
            print(f"{key}: {param.requires_grad}")

        learning_rate = trial.suggest_float("learning_rate", low=2e-5, high=5e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", low=0.0, high=0.0)
        adam_beta1 = trial.suggest_float("adam_beta1", low=0.9, high=0.9)
        adam_beta2 = trial.suggest_float("adam_beta2", low=0.999, high=0.999)
        adam_epsilon = trial.suggest_float("adam_epsilon", low=1e-8, high=1e-6, log=True)

        config = TrainingArguments(
            output_dir=f"./results/livedoor/trial_{trial.number:03d}",
            logging_dir=f"./logs/livedoor/trial_{trial.number:03d}",
            metric_for_best_model="f1_score",
            greater_is_better=True,
            save_total_limit=3,
            num_train_epochs=50,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
        )
        logger.info("Created training config")

        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=5),
        ]

        trainer = Trainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=metrics,
            callbacks=callbacks,
        )
        trainer.train()

        # TODO: get best metrics on eval_dataset during training
        #
        #       We may need to use callback...
        #       [x] history = trainer.train()
        #       [x] history #=> TrainOutput(global_step=xxx, training_loss=xxx, metrics={})
        #                                                                       ^^^^^^^^^^
        #                                                               doesn't contain eval metrics
        #
        state = trainer.evaluate()
        return state["eval_f1_score"]
    return objective


def metrics(p: EvalPrediction) -> Dict[str, float]:
    y_true = p.label_ids
    y_pred = p.predictions.argmax(axis=1)
    average = "weighted"

    print(classification_report(y_true, y_pred))
    return {
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1_score": f1_score(y_true, y_pred, average=average),
        "accuracy": accuracy_score(y_true, y_pred),
    }


if __name__ == "__main__":
    # model_name = "cl-tohoku/bert-large-japanese"
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    logger.info("Created tokenizer")

    items = read_livedoor("./data/livedoor")
    label_ids = [item["label_id"] for item in items]
    train_items, valid_items = train_test_split(items, stratify=label_ids, test_size=0.2)
    train_texts, train_labels = zip(*[(item["text"], item["label_id"]) for item in train_items])
    valid_texts, valid_labels = zip(*[(item["text"], item["label_id"]) for item in valid_items])
    train_texts = tokenizer(train_texts, padding=True, truncation=True, max_length=256)
    valid_texts = tokenizer(valid_texts, padding=True, truncation=True, max_length=256)
    train_dataset = LivedoorDataset(train_texts, labels=train_labels)
    valid_dataset = LivedoorDataset(valid_texts, labels=valid_labels)
    logger.info("Loaded dataset for training")

    num_labels = len(set(label_ids))
    objective = create_objective(model_name=model_name, num_labels=num_labels)
    study = create_study(storage=None, study_name="transformer-sandbox", direction="maximize")
    study.optimize(objective, n_trials=15)
