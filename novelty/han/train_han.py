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
    load_han_attn_encoder,
    load_han_bilstm_encoder,
    reset_model,
)
from utils.save_models import save_model, save_model_neptune

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
    parser.add_argument(
        "--use_nltk", action="store_true", help="Dataset imdb", default=False
    )
    parser.add_argument("--save", action="store_true", help="Save model")
    args = parser.parse_args()

    use_nltk = args.use_nltk
    seed_torch()

    if args.encoder == "bilstm":
        model_id = "DOC-5"
        encoder, Lang = load_han_bilstm_encoder(model_id)
    elif args.encoder == "attention":
        # model_id = "DOC-2"
        model_id = "DOC-13"
        encoder, Lang = load_han_attn_encoder(model_id)

    if args.webis:
        data_module = webis_data_module(Lang, use_nltk=use_nltk)
    elif args.dlnd:
        data_module = dlnd_data_module(Lang, use_nltk=use_nltk)
    elif args.apwsj:
        data_module = apwsj_data_module(Lang, use_nltk=use_nltk)

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
    expt_id = neptune_logger.experiment.id

    tensorboard_logger = TensorBoardLogger("lightning_logs")

    lr_logger = LearningRateLogger(logging_interval="step")
    neptune_logger.experiment.log_metric("epochs", EPOCHS)
    neptune_logger.experiment.log_text("Use NLTK", str(use_nltk))
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

    if args.save:
        model_data = {"model": model.model, "model_conf": model_conf, "Lang": Lang}
        save_path = save_model("han_novelty", expt_id, model_data)
        save_model_neptune(save_path, neptune_logger)
