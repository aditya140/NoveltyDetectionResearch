import sys

sys.path.append(".")
import joblib
import pickle
import argparse
from lang import *
from document.han.han import *
from snli.bilstm.bilstm import *
from snli.attn_enc.attn_enc import *
from joblib import Memory
import shutil
import pytorch_lightning as pl
from document.han.han import HAN_conf, HAN_classifier, HAN
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from utils.load_models import load_bilstm_encoder, load_attn_encoder
from utils.helpers import seed_torch
from utils.save_models import save_model, save_model_neptune
from document.train_utils import *
from datamodule import *
import os
from utils.keys import NEPTUNE_API


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty CNN Training")

    parser.add_argument("--save", action="store_true", help="Save model")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    parser.add_argument("--yelp", action="store_true", help="Dataset yelp")
    parser.add_argument("--imdb", action="store_true", help="Dataset imdb")

    args = parser.parse_args()

    seed_torch()

    if args.encoder == "bilstm":
        model_id = "SNLI-13"
        encoder, Lang = load_bilstm_encoder(model_id)
    elif args.encoder == "attention":
        model_id = "SNLI-12"
        encoder, Lang = load_attn_encoder(model_id)

    if args.imdb:
        data_module = imdb_data_module(Lang)
    elif args.yelp:
        data_module = yelp_data_module(Lang)

    params = {
        "optimizer_base": {
            "optim": "adamw",
            "lr": 0.0010039910781394373,
            "amsgrad": True,
            "scheduler": (
                "lambda",
                10,
                4,
            ),  # (Schedule type, after every ? epoch, Divide by a factor of)
        },
        "optimizer_tune": {
            "optim": "adagrad",
            "lr": 0.001,
            "weight_decay": 0.001,
            "scheduler": (
                "lambda",
                10,
                4,
            ),  # (Schedule type, after every ? epoch, Divide by a factor of)
        },
        "switch_epoch": 2,
    }

    model_conf = HAN_conf(100, encoder, **params)
    model = Document_model_clf(HAN_classifier, model_conf, params)

    EPOCHS = 6

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aparkhi/DocClassification",
        experiment_name="training",  # Optional,
        tags=[
            "HAN",
            "Training",
            ("Yelp" if args.yelp else ("IMDB" if args.imdb else "")),
        ],
    )
    expt_id = neptune_logger.experiment.id

    tensorboard_logger = TensorBoardLogger("lightning_logs")

    lr_logger = LearningRateLogger(logging_interval="step")
    neptune_logger.experiment.log_metric("epochs", EPOCHS)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=EPOCHS,
        progress_bar_refresh_rate=10,
        profiler=False,
        auto_lr_find=False,
        callbacks=[lr_logger, SwitchOptim()],
        logger=[neptune_logger, tensorboard_logger],
        row_log_interval=2,
    )
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)

    if args.save:
        model_data = {"model": model.model.han, "model_conf": model_conf, "Lang": Lang}
        save_path = save_model("han", expt_id, model_data)
        save_model_neptune(save_path, neptune_logger)
