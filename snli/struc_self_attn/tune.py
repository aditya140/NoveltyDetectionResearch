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
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
import json
import os
from snli.struc_self_attn.struc_self_attn_enc import *
from utils.keys import NEPTUNE_API
import argparse


def param_tuning(trial, conf, hparams):
    conf.num_layers = trial.suggest_int("num_layers", 1,3)
    conf.dropout = trial.suggest_float("dropout", 0.1, 0.5)
    conf.hidden_size = trial.suggest_categorical("hidden_size", [150, 300, 400, 500])
    conf.gated_embedding_dim = trial.suggest_categorical("gated_embedding_dim", [150, 300, 400, 500])
    conf.r = trial.suggest_categorical("r", [1, 5, 10, 15, 30 , 40])
    conf.freeze_embedding = trial.suggest_categorical("freeze_embedding", [True, False])
    conf.gated = trial.suggest_categorical("gated", [True, False])
    if conf.gated == False:
        conf.pool_strategy =  trial.suggest_categorical('pool_strategy',['max','avg'])
    conf.C = trial.suggest_categorical("C", [0, 0.2, 0.5, 0.8,1])
    conf.attention_layer_param = trial.suggest_categorical(
        "attention_param", [50, 100, 200, 300]
    )
    conf.fcs = trial.suggest_int("fcs", 0, 3)

    if conf.gated == False:
        conf.pool_strategy =  trial.suggest_categorical('pool_strategy',['max','avg'])
    print("Params selected")
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
        tags=["Hyperparams", "Structured Self Attention"],
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

    model_conf = Struc_Attn_Encoder_conf(
        Lang,
        embedding_matrix,
        **{"bidirectional": False, "num_layers": 1, "freeze_embedding": False, "batch_size":128,"max_len":100}
    )
    model = SNLI_struc_attn_model(
        Struc_Attn_encoder_snli,
        model_conf,
        hparams,
        trial_set={"trial_func": param_tuning, "trial": trial},
    )

    print(trial.params)
    neptune_logger.experiment.log_text("params", json.dumps(trial.params))

    trainer.fit(model, datamodule=data_module)

    return metrics_callback.metrics[-1]["val_checkpoint_on"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNLI Structured Self Attention Tuning")
    parser.add_argument("--glove", action="store_true", help="glove embedding")
    args = parser.parse_args()

    DIR = os.getcwd()
    MODEL_DIR = os.path.join(DIR, "result")
    EPOCHS = 2

    if args.glove:
        data_module = snli_glove_data_module(128)
        Lang = data_module.Lang
        embedding_matrix = read_embedding_file(Lang)
    else:
        data_module = snli_bert_data_module(128)
        Lang = data_module.Lang
        embedding_matrix = None

    hparams = {
        "optimizer_base":{
            "optim": "adamw",
            "lr": 0.0010039910781394373,
            "scheduler": "const"
            },
        "optimizer_tune":{
            "optim": "sgd",
            "lr": 3e-4,
            "momentum": 0,
            "weight_decay": 0,
            "scheduler": "const"
        },
        "switch_epoch":10,
    } 

    Lang = data_module.Lang

    pruner = optuna.pruners.NopPruner()
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=150)
