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

    args = parser.parse_args()

    if args.encoder == "bilstm":
        model_id = "SNLI-13"
        encoder, Lang = load_bilstm_encoder(model_id)
    elif args.encoder == "attention":
        model_id = "SNLI-12"
        encoder, Lang = load_attn_encoder(model_id)

    if args.webis:
        data_module = webis_data_module(Lang)
    elif args.dlnd:
        data_module = dlnd_data_module(Lang)
    elif args.apwsj:
        data_module = apwsj_data_module(Lang)

    # if args.webis:
    #     data_module = webis_crossval_datamodule(Lang)
    # elif args.dlnd:
    #     data_module = dlnd_crossval_datamodule(Lang)
    # elif args.apwsj:
    #     data_module = apwsj_crossval_datamodule(Lang)
    # data_module.set_fold(0)

    params = {
        "num_filters": 100,
        "encoder_dim": encoder.conf.hidden_size,
        "dropout": 0.3,
        "expand features": True,
        "filter_sizes": [4, 6, 9],
        "freeze_embedding": False,
        "activation": "tanh",
        "optim": "adamw",
        "weight_decay": 0.1,
        "lr": 0.00010869262115700171,
        "scheduler": "lambda",
    }

    model_conf = Novelty_CNN_conf(100, encoder, **params)
    model = Novelty_CNN_model(DeepNoveltyCNN, model_conf, params)

    EPOCHS = 5

    tensorboard_logger = TensorBoardLogger("lightning_logs")

    if args.log:
        neptune_logger = NeptuneLogger(
            api_key=NEPTUNE_API,
            project_name="aparkhi/Novelty",
            experiment_name="Evaluation",  # Optional,
            tags=[
                ("Webis" if args.webis else ("DLND" if args.dlnd else "APWSJ")),
                "test",
                "CNN",
            ],
        )
        expt_id = neptune_logger.experiment.id
        neptune_logger.experiment.log_metric("epochs", EPOCHS)
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

    model_data = {"model": model.model, "model_conf": model_conf, "Lang": Lang}
    save_path = save_model("cnn_novelty", expt_id, model_data)
    save_model_neptune(save_path, neptune_logger)
