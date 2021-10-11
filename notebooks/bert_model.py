# Load Dataset
import os
import sys

sys.path.append(os.path.abspath("../"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch

from sklearn.model_selection import train_test_split


from transformers import (
    DistilBertTokenizerFast,
    BertTokenizerFast,
    LongformerTokenizerFast,
    RobertaTokenizerFast,
)
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import BertForSequenceClassification
from transformers import LongformerForSequenceClassification
from transformers import RobertaForSequenceClassification


def load_dataset():
    # Load Dataset
    with open(
        "/content/NoveltyDetectionResearch/.data/dlnd/TAP-DLND-1.0_LREC2018_modified/dlnd.jsonl",
        "r",
    ) as f:
        data = f.readlines()
        dataset = [json.loads(line) for line in data]

    texts = [(i["source"], i["target_text"]) for i in dataset]
    labels = [1 if i["DLA"] == "Novel" else 0 for i in dataset]
    return texts, labels


texts, labels = load_dataset()
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2
)


from spacy.lang.en import English  # updated

nlp = English()
nlp.add_pipe(nlp.create_pipe("sentencizer"))  # updated


def split_sentences(text):
    doc = nlp(text.strip().replace("\n", " . "))
    sentences = list(
        filter(lambda x: x != ".", [sent.string.strip() for sent in doc.sents])
    )
    return sentences


# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


def process_text(texts):
    texts_new = []
    for i in train_texts:
        new_i = (
            tokenizer(split_sentences(i[0]), truncation=False, padding=True),
            tokenizer(split_sentences(i[1]), truncation=False, padding=True),
        )
        texts_new.append(new_i)
    return texts_new


train_encodings = process_text(train_texts)
val_encodings = process_text(val_texts)
test_encodings = process_text(test_texts)


class DLNDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = DLNDDataset(train_encodings, train_labels)
val_dataset = DLNDDataset(val_encodings, val_labels)
test_dataset = DLNDDataset(test_encodings, test_labels)


import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1_novel": f1[0],
        "P_novel": precision[0],
        "R_novel": recall[0],
        "f1_non_novel": f1[1],
        "P_non_novel": precision[1],
        "R_non_novel": recall[1],
    }


training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=15,  # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=100,  # number of steps between logging
    evaluation_strategy="steps",  # evaluation strategy
    eval_steps=200,  # number of steps between evaluations
    gradient_accumulation_steps=2,  # number of steps for gradient accumulation
)

# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
# model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
model = RobertaForSequenceClassification.from_pretrained("roberta-base")


trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,  # function to compute metrics
)

trainer.train()

trainer.evaluate()


trainer.evaluate(test_dataset)
