import sys

# Disable tensorflow warning
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

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
import seaborn as sns
import os
from utils.keys import NEPTUNE_API
import warnings


# remove pytorch lightning logs
import logging

logging.getLogger("lightning").setLevel(0)
warnings.filterwarnings("ignore")


def train_model(args):

    tp = args.train_samples

    if args.encoder == "bilstm":
        encoder, Lang = load_bilstm_encoder()
    elif args.encoder == "attention":
        encoder, Lang = load_attn_encoder()
    data_module = (
        webis_data_module(Lang, tp, seed=args.seed)
        if args.webis
        else dlnd_data_module(Lang, tp, seed=args.seed)
    )

    params = {
        "num_filters": 60,
        "dropout": 0.3,
        "expand features": False,
        "filter_sizes": [4, 6, 9],
        "freeze_embedding": True,
        "activation": "tanh",
        "optim": "adamw",
        "weight_decay": 0.1,
        "lr": 0.00010869262115700171,
        "scheduler": "lambda",
    }

    model_conf = Novelty_CNN_conf(100, encoder, **params)
    model = Novelty_CNN_model(DeepNoveltyCNN, model_conf, params)

    EPOCHS = args.epochs

    tensorboard_logger = TensorBoardLogger("lightning_logs")

    lr_logger = LearningRateLogger(logging_interval="step")
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=EPOCHS,
        profiler=False,
        auto_lr_find=False,
        callbacks=[lr_logger],
        logger=[tensorboard_logger],
        row_log_interval=2,
        weights_summary=None,
        progress_bar_refresh_rate=0,
        val_percent_check=0,
    )
    trainer.fit(model, data_module)
    test_res = trainer.test(model, datamodule=data_module, verbose=False)
    return test_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty CNN Training")

    parser.add_argument("--webis", action="store_true", help="Webis dataset")
    parser.add_argument("--dlnd", action="store_true", help="DLND dataset")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    parser.add_argument(
        "--train_samples", type=int, help="Training Samples", default=150
    )
    parser.add_argument("--epochs", type=int, help="Epochs", default=5)
    args = parser.parse_args()

    res = train_model(args)
    print(res)
