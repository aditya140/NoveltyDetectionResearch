import sys

sys.path.append(".")
import joblib
import pickle
import argparse
from lang import *
from novelty.han.han_novelty import *
from snli.bilstm.bilstm import *
from snli.attn_enc.attn_enc import *
from joblib import Memory
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from utils.load_models import (
    load_bilstm_encoder,
    load_attn_encoder,
    load_han_clf_encoder,
    load_han_reg_encoder,
    reset_model,
)
from utils.helpers import seed_torch
from novelty.train_utils import *
from datamodule import *
import os
from utils.keys import NEPTUNE_API


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty HAN Training")

    parser.add_argument("--webis", action="store_true", help="Webis dataset")
    parser.add_argument("--dlnd", action="store_true", help="DLND dataset")
    parser.add_argument("--apwsj", action="store_true", help="APWSJ dataset")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    parser.add_argument(
        "--reset", action="store_true", help="Reset Weights", default=False
    )

    args = parser.parse_args()

    seed_torch()

    if args.encoder == "reg":
        model_id = "DOC-4"
        encoder, Lang = load_han_reg_encoder(model_id)
    elif args.encoder == "clf":
        model_id = "DOC-2"
        encoder, Lang = load_han_clf_encoder(model_id)

    if args.webis:
        data_module = webis_data_module(Lang)
    elif args.dlnd:
        data_module = dlnd_data_module(Lang)
    elif args.apwsj:
        data_module = apwsj_data_module(Lang)

    params = {
        "optim": "adamw",
        "weight_decay": 0.1,
        "lr": 0.00010869262115700171,
        "scheduler": "lambda",
    }

    model_conf = HAN_Novelty_conf(encoder, **params)
    model = Novelty_CNN_model(HAN_Novelty, model_conf, params)

    if args.reset:
        print("Reinitializing weights")
        model.model = reset_model(model.model)

    EPOCHS = 4

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aparkhi/Novelty",
        experiment_name="Evaluation",  # Optional,
        tags=[
            ("Webis" if args.webis else ("DLND" if args.dlnd else "APWSJ")),
            "test",
            "HAN",
            "encoder_" + args.encoder,
            ("weights_reset" if args.reset else "pretrained"),
        ],
    )

    tensorboard_logger = TensorBoardLogger("lightning_logs")

    lr_logger = LearningRateLogger(logging_interval="step")
    neptune_logger.experiment.log_metric("epochs", EPOCHS)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=EPOCHS,
        progress_bar_refresh_rate=10,
        profiler=False,
        auto_lr_find=False,
        callbacks=[lr_logger],
        logger=[neptune_logger, tensorboard_logger],
        row_log_interval=2,
    )
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)
