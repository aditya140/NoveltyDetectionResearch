import sys

sys.path.append(".")

import pytorch_lightning as pl
import torch.nn.functional as F
from utils import *
from datamodule import *
from pytorch_lightning.callbacks import LearningRateLogger
from snli.diin.diin import *
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.metrics import Accuracy
import pickle
import os
from joblib import Memory
import shutil
import argparse
from lang import *
from snli.train_utils import SNLI_model, snli_glove_data_module, snli_bert_data_module
from utils.keys import NEPTUNE_API

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNLI DIIN Training")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--save", action="store_true", help="Save model")
    parser.add_argument("--glove", action="store_true", help="glove embedding")
    parser.add_argument("--batch_size", default=70, type=int, help="batch size")

    args = parser.parse_args()

    if args.glove:
        data_module = snli_glove_data_module(args.batch_size)
        Lang = data_module.Lang
        embedding_matrix = read_embedding_file(Lang)
    else:
        data_module = snli_bert_data_module(args.batch_size)
        Lang = data_module.Lang
        embedding_matrix = None

    conf_kwargs = {
        "num_layers": 2,
        "dropout": 0.10018262692246818,
        "hidden_size": 300,
        "bidirectional": True,
        "freeze_embedding": False,
        "activation": "tanh",
        "fcs": 0,
        "glove": args.glove,
    }

    hparams = {
        "optim": "adamw",
        "lr": 0.0010039910781394373,
        "scheduler": "lambda",
    }  # "momentum":0.9}
    # hparams = {"optim": "adadelta", "lr": 0.5,"rho":0.9,"eps":1e-8,"scheduler":'lambda',"weight_decay":0}  # "momentum":0.9}
    Lang = data_module.Lang

    model_conf = DIIN_conf(Lang, embedding_matrix, **conf_kwargs)
    model = SNLI_model(DIIN, model_conf, hparams)

    EPOCHS = 10

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aditya140/SNLI",
        experiment_name="Evaluation",
        tags=["DIIN", args.tag],
    )
    tensorboard_logger = TensorBoardLogger("lightning_logs")
    lr_logger = LearningRateLogger(logging_interval="step")
    profiler = AdvancedProfiler()

    neptune_logger.experiment.log_metric("epochs", EPOCHS)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=EPOCHS,
        progress_bar_refresh_rate=10,
        # profiler=profiler,
        auto_lr_find=False,
        callbacks=[lr_logger],
        logger=[neptune_logger, tensorboard_logger],
        row_log_interval=2,
        benchmark=True,
    )
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)
