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
from document.han.han import HAN_conf,HAN_classifier,HAN
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from utils import load_bilstm_encoder, load_attn_encoder, seed_torch
from document.train_utils import *
from datamodule import *
import os
from keys import NEPTUNE_API


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty CNN Training")

    parser.add_argument("--save", action="store_true", help="Save model")
    parser.add_argument("--encoder", type=str, help="Encoder Type")

    args = parser.parse_args()

    seed_torch()

    if args.encoder == "bilstm":
        encoder, Lang = load_bilstm_encoder()
    elif args.encoder == "attention":
        encoder, Lang = load_attn_encoder()
    data_module = imdb_data_module(Lang)

    params = {
        "optimizer_base":{
            "optim": "adamw",
            "lr": 0.0010039910781394373,
            "amsgrad":True,
            "scheduler": ("lambda",10,4)  # (Schedule type, after every ? epoch, Divide by a factor of)
            },
        "optimizer_tune":{
            "optim": "adagrad",
            "lr": 0.001,
            "weight_decay": 0.001,
            "scheduler": ("lambda",10,4)  # (Schedule type, after every ? epoch, Divide by a factor of)
        },
        "switch_epoch":2,
    } 

    model_conf = HAN_conf(100, encoder, **params)
    model = Document_model_clf(HAN_classifier, model_conf, params)

    EPOCHS = 15

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aditya140/Imdb",
        experiment_name="training",  # Optional,
        tags=['HAN'],
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
        callbacks=[lr_logger,SwitchOptim()],
        logger=[neptune_logger, tensorboard_logger],
        row_log_interval=2,
    )
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)

    if args.save:
        MODEL_PATH = "./models/document_imdb_han_clf/"
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        torch.save(model.model.han.state_dict(), MODEL_PATH + "weights.pt")
        with open(MODEL_PATH + "model_conf.pkl", "wb") as f:
            pickle.dump(model_conf, f)
        with open(MODEL_PATH + "lang.pkl", "wb") as f:
            pickle.dump(Lang, f)
        shutil.make_archive("./models/document_imdb_han_clf", "zip", "./models/document_imdb_han_clf")
        neptune_logger.experiment.log_artifact("./models/document_imdb_han_clf.zip")
