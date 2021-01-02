import sys

sys.path.append(".")

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback
import json
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
import os
import pytorch_lightning as pl
import torch.nn.functional as F
from utils import *
from datamodule import *
from utils.load_models import load_bilstm_encoder, load_attn_encoder
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
import json
import os
from novelty.dan.train_dan import *
from novelty.dan.dan import *


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


encoder, Lang = load_encoder("bilstm")
data_module = webis_data_module(Lang)

DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")
EPOCHS = 2
hparams = {"optim": "adam", "lr": 1e-3}  # "momentum":0.9}


def trial_init(self, trial):
    self.conf.num_filters = trial.suggest_int("num_filters", 50, 100)
    self.conf.dropout = trial.suggest_float("dropout", 0.1, 0.5)
    self.conf.expand_features = trial.suggest_categorical(
        "expand features", [True, False]
    )
    self.conf.filter_sizes = trial.suggest_categorical(
        "filter_sizes",
        [
            (3, 4, 5),
            (4, 5, 6),
            (5, 6, 7),
            (3, 5, 7),
            (3, 6, 9),
            (3, 4, 5, 6, 7, 8),
            (4, 6, 9, 12),
        ],
    )
    self.conf.freeze_encoder = trial.suggest_categorical(
        "freeze_embedding", [True, False]
    )
    self.conf.activation = trial.suggest_categorical(
        "activation", ["tanh", "relu", "leakyrelu"]
    )
    self.hparams.optim = trial.suggest_categorical("optimizer", ["adamw"])
    if self.hparams.optim == "adam":
        self.hparams.lr = trial.suggest_loguniform("lr", 1e-7, 0.1)
        self.hparams.weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.9)
    if self.hparams.optim == "adamw":
        self.hparams.lr = trial.suggest_loguniform("lr", 1e-7, 0.1)
    if self.hparams.optim == "sgd":
        self.hparams.lr = trial.suggest_loguniform("lr", 1e-5, 1)
        self.hparams.momentum = trial.suggest_float("momentum", 0.01, 0.9)
        self.hparams.weight_decay = trial.suggest_float("weight_decay", 0.01, 0.9)


def objective(trial):
    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number)),
        monitor="val_checkpoint_on",
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    metrics_callback = MetricsCallback()

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZGZjOGZjMTEtNzk2MS00NzllLTkxOTAtMjI3NzU4MzE2YmM3In0=",
        project_name="aparkhi/Novelty",
        experiment_name="Hyperparameter Tuning",  # Optional,
        tags=[
            "Hyperparams",
            "CNN",
        ],
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

    model_conf = Novelty_CNN_conf(100, encoder)

    model = Novelty_CNN_model(DeepNoveltyCNN, model_conf, params)

    neptune_logger.experiment.log_text("params", json.dumps(trial.params))

    trainer.fit(model, datamodule=data_module)

    return metrics_callback.metrics[-1]["val_checkpoint_on"].item()


pruner = optuna.pruners.NopPruner()
pruner = optuna.pruners.HyperbandPruner()

study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=150)


print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
