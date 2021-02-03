import sys

sys.path.append(".")
import joblib
import pickle
import argparse
from lang import *
from novelty.cnn.cnn_model import *
from novelty.diin.diin import *

from snli.bilstm.bilstm import *
from snli.attn_enc.attn_enc import *
from joblib import Memory
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from utils.load_models import load_bilstm_encoder, load_attn_encoder
from novelty.train_utils import *
from datamodule import *
import os
from utils.keys import NEPTUNE_API


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty CNN Training")

    parser.add_argument(
        "--webis", action="store_true", help="Webis dataset", default=False
    )
    parser.add_argument(
        "--dlnd", action="store_true", help="DLND dataset", default=False
    )
    parser.add_argument("--apwsj", action="store_true", help="apwsj dataset")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    parser.add_argument("--batch_size", type=int, help="batch size", default=25)
    parser.add_argument("--use_nltk", action="store_true", help="Dataset imdb", default=False)

    args = parser.parse_args()

    use_nltk=args.use_nltk

    if args.encoder == "bilstm":
        model_id = "SNLI-13"
        encoder, Lang = load_bilstm_encoder(model_id)
    elif args.encoder == "attention":
        model_id = "SNLI-12"
        encoder, Lang = load_attn_encoder(model_id)

    if args.webis:
        data_module = webis_data_module(Lang,use_nltk=use_nltk)
    elif args.dlnd:
        data_module = dlnd_data_module(Lang,use_nltk=use_nltk)
    elif args.apwsj:
        data_module = apwsj_data_module(Lang,use_nltk=use_nltk)

        
    data_module.batch_size = args.batch_size

    params = {
        "optim": "adamw",
        "lr": 0.00060869262115700171,
        "weight_decay": 0.01,
        "scheduler": "constant",
        "encoder_type": args.encoder,
        "batch_size": args.batch_size,
        "embedding_dim": encoder.conf.hidden_size,
        "analysis":True,
        "analysisFile":'diin_analysis.csv'
    }

    model_conf = DIIN_conf(100, encoder, **params)
    model = Novelty_model(DIIN, model_conf, params)

    EPOCHS = 3

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aparkhi/Novelty",
        tags=[
            ("Webis" if args.webis else ("DLND" if args.dlnd else "APWSJ")),
            "test",
            "DIIN",
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
