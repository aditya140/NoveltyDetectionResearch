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
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
import json
import os
from novelty.diin.train_diin import *
from novelty.diin.diin import *
from utils.keys import NEPTUNE_API

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def trial_init(trial,conf,hparams):
    conf.hidden_size = trial.suggest_int("hidden_size", 200,800 )
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    conf.dropout = [dropout]*5
    conf.num_layers = trial.suggest_int("num_layers", 1,3)
    conf.dense_net_layers = trial.suggest_int("dense_net_layers", 1,5)
    conf.dense_net_transition_rate = trial.suggest_uniform("dense_net_transition_rate", 0.1,0.7)
    conf.dense_net_first_scale_down_ratio = trial.suggest_uniform("dense_net_first_scale_down_ratio", 0.1,0.7)
    conf.dense_net_channels = trial.suggest_int("dense_net_channels", 50,200)
    conf.dense_net_channels = trial.suggest_int("dense_net_channels", 50,200)
    
    # self.hparams.optim = trial.suggest_categorical("optimizer", ["adamw"])
    # if self.hparams.optim == "adam":
    #     self.hparams.lr = trial.suggest_loguniform("lr", 1e-7, 0.1)
    #     self.hparams.weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.2)
    # if self.hparams.optim == "adamw":
    #     self.hparams.lr = trial.suggest_loguniform("lr", 1e-7, 0.1)
    #     self.hparams.weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.2)
    # if self.hparams.optim == "sgd":
    #     self.hparams.lr = trial.suggest_loguniform("lr", 1e-5, 1)
    #     self.hparams.momentum = trial.suggest_float("momentum", 0.01, 0.9)
    #     self.hparams.weight_decay = trial.suggest_float("weight_decay", 0.01, 0.2)
    return conf,hparams

def objective(trial):

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number)),
        monitor="val_checkpoint_on",
    )

    metrics_callback = MetricsCallback()

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aditya140/Novelty",
        experiment_name="Hyperparameter Tuning",  # Optional,
        tags=[
            "Hyperparams",
            "DIIN",
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
        accumulate_grad_batches=3
    )

    model_conf = DIIN_conf(100, encoder)


    params = {
        "optim": "adamw",
        "lr": 0.00060869262115700171,
        "weight_decay":0.01,
        "scheduler": "constant",
        "encoder_type": args.encoder,
        "batch_size": args.batch_size,
    }
    trial_set = {"trial_func":trial_init,"trial":trial}
    model = Novelty_CNN_model(DIIN, model_conf, params,trial_set)
    neptune_logger.experiment.log_text("trial_params", json.dumps(trial.params))
    # neptune_logger.experiment.log_text("params", json.dumps(vars(model_conf)))

    trainer.fit(model, datamodule=data_module)
    return metrics_callback.metrics[-1]["val_checkpoint_on"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIIN Novelty Hyperparam Tuning")
    parser.add_argument(
        "--webis", action="store_true", help="Webis dataset", default=False
    )
    parser.add_argument(
        "--dlnd", action="store_true", help="DLND dataset", default=False
    )
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    parser.add_argument("--batch_size", type=int, help="batch size", default=25)
    args = parser.parse_args()

    DIR = os.getcwd()
    MODEL_DIR = os.path.join(DIR, "result")
    EPOCHS = 1



    if args.encoder == "bilstm":
        encoder, Lang = load_bilstm_encoder()
    elif args.encoder == "attention":
        encoder, Lang = load_attn_encoder()

    data_module = webis_data_module(Lang) if args.webis else dlnd_data_module(Lang)
    data_module.batch_size = args.batch_size

    DIR = os.getcwd()
    MODEL_DIR = os.path.join(DIR, "result")
    EPOCHS = 5
    

    pruner = optuna.pruners.NopPruner()
    # pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=300)
