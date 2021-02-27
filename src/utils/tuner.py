import abc
import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Tuner(abc.ABC):
    def __init__(self, dataset_conf, model_conf, hparams, **kwargs):
        self.dataset_conf = dataset_conf
        self.model_conf = model_conf
        self.hparams = hparams
        self.init_kwargs = kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abc.abstractmethod
    def load_model(self, model_conf, **kwargs):
        """
        Implementation on the process of loading dataset.
        """


def validate_model(model, optimizer, criterion, val_iterator, **kwargs):
    model.eval()
    if hasattr(val_iterator, "init_epoch") and callable(val_iterator.init_epoch):
        val_iterator.init_epoch()
    n_correct, n_total, n_loss = 0, 0, 0

    if kwargs.get("batch_attr", None) == None:
        raise ValueError(
            """Please provide batch attributes which need to be passed to the model (eg, model inputs, labels)
        batch_attr={model_inp:[source,target],label:label}"""
        )
    else:
        batch_attr = kwargs["batch_attr"]
        input_attr = batch_attr["model_inp"]
        label_attr = batch_attr["label"]

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_iterator):
            model_inp = [getattr(batch, i) for i in input_attr]
            label = getattr(batch, label_attr)
            batch_size = label.shape[0]

            answer = model(*model_inp)
            loss = criterion(answer, label)
            n_correct += (
                (torch.max(F.softmax(answer, dim=1), 1)[1].view(label.size()) == label)
                .sum()
                .item()
            )
            n_total += batch_size
            n_loss += loss.item()

        val_loss = n_loss / n_total
        val_acc = 100.0 * n_correct / n_total
        return val_loss, val_acc


def train_model(model, optimizer, criterion, train_iterator, **kwargs):
    model.train()

    if hasattr(train_iterator, "init_epoch") and callable(train_iterator.init_epoch):
        train_iterator.init_epoch()

    n_correct, n_total, n_loss = 0, 0, 0
    if kwargs.get("batch_attr", None) == None:
        raise ValueError(
            """Please provide batch attributes which need to be passed to the model (eg, model inputs, labels)
        batch_attr={model_inp:[source,target],label:label}"""
        )
    else:
        batch_attr = kwargs["batch_attr"]
        input_attr = batch_attr["model_inp"]
        label_attr = batch_attr["label"]

    for batch_idx, batch in enumerate(train_iterator):
        optimizer.zero_grad()
        model_inp = [getattr(batch, i) for i in input_attr]
        label = getattr(batch, label_attr)
        batch_size = label.shape[0]

        answer = model(*model_inp)
        loss = criterion(answer, label)
        n_correct += (
            (torch.max(F.softmax(answer, dim=1), 1)[1].view(label.size()) == label)
            .sum()
            .item()
        )
        n_total += batch_size
        n_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = n_loss / n_total
    train_acc = 100.0 * n_correct / n_total
    # print("Train Accuracy", train_acc)
    return train_loss, train_acc
