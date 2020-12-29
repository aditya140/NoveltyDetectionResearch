import sys

sys.path.append(".")
import joblib
import pickle
import argparse
from lang import *
from novelty.cnn.cnn_model import *
from snli.bilstm.bilstm import *
from snli.attn_enc.attn_enc import *
from joblib import Memory
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from utils import load_bilstm_encoder, load_attn_encoder
from GAN.train_utils import *
from datamodule import *
import os
from keys import NEPTUNE_API


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty CNN Training")

    parser.add_argument("--webis", action="store_true", help="Webis dataset")
    parser.add_argument("--dlnd", action="store_true", help="DLND dataset")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    args = parser.parse_args()


    train_percentage = [0.01,0.05,0.1, 0.2, 0.5, 0.8,0.9]
    for tp in train_percentage:
        if args.encoder == "bilstm":
            encoder, Lang = load_bilstm_encoder()
        elif args.encoder == "attention":
            encoder, Lang = load_attn_encoder()
        data_module = webis_data_module(Lang,tp) if args.webis else dlnd_data_module(Lang,tp)

        params = {
            "num_filters": 60,
            "dropout": 0.3,
            "expand features": False,
            "filter_sizes": [4, 6, 9],
            "freeze_embedding": True,
            "activation": "tanh",
            "optim": "adamw",
            "weight_decay":0.1,
            "lr": 0.00010869262115700171,
            "scheduler": "lambda",
        }

        model_conf = Novelty_CNN_conf(100, encoder, **params)
        model = Novelty_CNN_model(DeepNoveltyCNN, model_conf, params)

        EPOCHS = 4

        tensorboard_logger = TensorBoardLogger("lightning_logs")

        lr_logger = LearningRateLogger(logging_interval="step")
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=EPOCHS,
            progress_bar_refresh_rate=10,
            profiler=False,
            auto_lr_find=False,
            callbacks=[lr_logger],
            logger=[ tensorboard_logger],
            row_log_interval=2,
        )
        trainer.fit(model, data_module)
        trainer.test(model, datamodule=data_module)
