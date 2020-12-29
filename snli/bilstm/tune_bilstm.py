import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback
import sys

sys.path.append(".")

from snli.train_utils import *


import pytorch_lightning as pl
import torch.nn.functional as F
from utils import *
from datamodule import *
from pytorch_lightning.callbacks import LearningRateLogger
from snli.bilstm.bilstm import *
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
import json
import os
from utils.keys import NEPTUNE_API
import argparse


def param_tuning(trial, conf, hparams):
    conf.num_layers = trial.suggest_int("num_layers", 1, 4)
    conf.dropout = trial.suggest_float("dropout", 0.1, 0.5)
    conf.hidden_size = trial.suggest_categorical("hidden_size", [150, 300, 400, 500])
    conf.bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    conf.freeze_embedding = trial.suggest_categorical("freeze_embedding", [True, False])
    conf.activation = trial.suggest_categorical(
        "activation", ["tanh", "relu", "leakyrelu"]
    )
    conf.fcs = trial.suggest_int("fcs", 0, 3)
    hparams.optim = trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd"])
    if hparams.optim == "adam":
        hparams.lr = trial.suggest_loguniform("lr", 1e-7, 0.1)
        hparams.weight_decay = trial.suggest_float("weight_decay", 0.1, 0.9)
    if hparams.optim == "adamw":
        hparams.lr = trial.suggest_loguniform("lr", 1e-7, 0.1)
    if hparams.optim == "sgd":
        hparams.lr = trial.suggest_loguniform("lr", 1e-5, 1)
        hparams.momentum = trial.suggest_float("momentum", 0.1, 0.9)
        hparams.weight_decay = trial.suggest_float("weight_decay", 0.1, 0.9)
    return conf, hparams


def objective(trial):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number)),
        monitor="val_checkpoint_on",
    )

    metrics_callback = MetricsCallback()

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aditya140/SNLI",
        experiment_name="Hyperparameter Tuning",  # Optional,
        tags=["Hyperparams", "Bilstm"],
    )

    lr_logger = LearningRateLogger(logging_interval="step")

    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        max_epochs=EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        early_stop_callback=PyTorchLightningPruningCallback(
            trial, monitor="val_checkpoint_on"
        ),
        progress_bar_refresh_rate=50,
        logger=[neptune_logger],
        row_log_interval=2,
    )

    model_conf = Bi_LSTM_Encoder_conf(
        Lang,
        embedding_matrix,
        **{"bidirectional": False, "num_layers": 1, "freeze_embedding": False}
    )
    print(param_tuning)
    model = SNLI_model(
        Bi_LSTM,
        model_conf,
        hparams,
        trial_set={"trial_func": param_tuning, "trial": trial},
    )
    trainer.fit(model, datamodule=data_module)
    return metrics_callback.metrics[-1]["val_checkpoint_on"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNLI Bi Lstm Tuning")
    parser.add_argument("--glove", action="store_true", help="glove embedding")
    args = parser.parse_args()

    DIR = os.getcwd()
    MODEL_DIR = os.path.join(DIR, "result")
    EPOCHS = 1

    if args.glove:
        data_module = snli_glove_data_module(128)
        Lang = data_module.Lang
        embedding_matrix = read_embedding_file(Lang)
    else:
        data_module = snli_bert_data_module(128)
        Lang = data_module.Lang
        embedding_matrix = None

    hparams = {"optim": "adam", "lr": 1e-3, "scheduler": "lambda"}  # "momentum":0.9}
    Lang = data_module.Lang

    pruner = optuna.pruners.NopPruner()
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=150)
