import sys

sys.path.append(".")
import joblib
import pickle
import argparse
from lang import *
from novelty.adin.adin import *
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
from utils.save_models import save_model, save_model_neptune
from novelty.train_utils import *
from datamodule import *
import os
from utils.keys import NEPTUNE_API


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty CNN Training")

    parser.add_argument("--webis", action="store_true", help="Webis dataset")
    parser.add_argument("--dlnd", action="store_true", help="DLND dataset")
    parser.add_argument("--apwsj", action="store_true", help="apwsj dataset")
    parser.add_argument("--save", action="store_true", help="Save model")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    parser.add_argument("--log", action="store_true", help="Log to neptune")
    parser.add_argument(
        "--use_nltk", action="store_true", help="Dataset imdb", default=False
    )

    parser.add_argument("--hidden_size", type=int, help="Hidden Size")
    parser.add_argument("--N", type=int, help="N")
    parser.add_argument("--k", type=int, help="k")
    parser.add_argument("--num_layers", type=int, help="num_layers")
    parser.add_argument(
        "--scheduler", type=str, help="scheduler type (lambda or constant or plateau)"
    )

    args = parser.parse_args()

    use_nltk = args.use_nltk

    if args.encoder == "bilstm":
        model_id = "SNLI-13"
        encoder, Lang = load_bilstm_encoder(model_id)
    elif args.encoder == "attention":
        model_id = "SNLI-12"
        encoder, Lang = load_attn_encoder(model_id)

    if args.webis:
        data_module = webis_data_module(Lang, use_nltk=use_nltk)
    elif args.dlnd:
        data_module = dlnd_data_module(Lang, use_nltk=use_nltk)
    elif args.apwsj:
        data_module = apwsj_data_module(Lang, use_nltk=use_nltk)

    params = {
        "encoder_dim": encoder.conf.hidden_size,
        "optim": "adamw",
        "weight_decay": 0.1,
        "lr": 0.00010869262115700171,
        "scheduler": "lambda",
    }
    if args.hidden_size != None:
        params["hidden_size"] = args.hidden_size

    if args.N != None:
        params["N"] = args.N

    if args.k != None:
        params["k"] = args.k

    if args.num_layers != None:
        params["num_layers"] = args.num_layers

    if args.scheduler != None:
        params["scheduler"] = args.scheduler

    model_conf = ADIN_conf(100, encoder, **params)
    model = Novelty_CNN_model(ADIN, model_conf, params)

    EPOCHS = 10

    tensorboard_logger = TensorBoardLogger("lightning_logs")

    if args.log:
        neptune_logger = NeptuneLogger(
            api_key=NEPTUNE_API,
            project_name="aparkhi/Novelty",
            experiment_name="Evaluation",  # Optional,
            tags=[
                ("Webis" if args.webis else ("DLND" if args.dlnd else "APWSJ")),
                "ADIN",
            ],
        )
        expt_id = neptune_logger.experiment.id
        neptune_logger.experiment.log_metric("epochs", EPOCHS)
        neptune_logger.experiment.log_text("Use NLTK", str(use_nltk))
        loggers = [neptune_logger, tensorboard_logger]
    else:
        loggers = [tensorboard_logger]

    lr_logger = LearningRateLogger(logging_interval="step")
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=EPOCHS,
        progress_bar_refresh_rate=10,
        profiler=False,
        auto_lr_find=False,
        callbacks=[lr_logger],
        logger=loggers,
        row_log_interval=2,
    )
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)
