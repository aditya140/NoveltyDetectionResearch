import sys

sys.path.append(".")

import pytorch_lightning as pl
import torch.nn.functional as F
from datamodule import *
from pytorch_lightning.callbacks import LearningRateLogger
from snli.attn_enc.attn_enc import *
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.metrics import Accuracy
import pickle
import os
import joblib
import shutil
import argparse
from lang import *
from snli.train_utils import SNLI_model, snli_glove_data_module, snli_bert_data_module,SwitchOptim
from utils.keys import NEPTUNE_API
from utils.helpers import seed_torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNLI Attention BiLSTM Training")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--save", action="store_true", help="Save model")
    parser.add_argument("--glove", action="store_true", help="glove embedding")
    args = parser.parse_args()

    seed_torch()

    if args.glove:
        data_module = snli_glove_data_module(128)
        Lang = data_module.Lang
        embedding_matrix = read_embedding_file(Lang)
    else:
        data_module = snli_bert_data_module(128)
        Lang = data_module.Lang
        embedding_matrix = None

    conf_kwargs = {
        "num_layers": 2,
        "dropout": 0.10018262692246818,
        "embedding_dim":300,
        "hidden_size": 300,
        "attention_layer_param":200,
        "bidirectional": True,
        "freeze_embedding": False,
        "activation": "tanh",
        "fcs": 0,
        "glove": args.glove,
        "batch_size": 128,
        "max_len": 100,
    }

    hparams = {
        "optimizer_base":{
            "optim": "adamw",
            "lr": 0.0010039910781394373,
            "scheduler": "const"
            },
        "optimizer_tune":{
            "optim": "adadelta",
            "lr": 3e-4,
            "rho": 0.9,
            "eps":1e-5,
            "weight_decay": 0.1,
            "scheduler": "const"
        },
        "switch_epoch":5,
    } 

    Lang = data_module.Lang

    model_conf = Attn_Encoder_conf(Lang, embedding_matrix, **conf_kwargs)
    model = SNLI_model(Attn_encoder_snli, model_conf, hparams)

    EPOCHS = 10

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aparkhi/SNLI",
        experiment_name="Evaluation",
        tags=["Attention", args.tag ],
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
        if not os.path.exists("models/attn_encoder"):
            os.makedirs("models/attn_encoder")
        MODEL_PATH = "./models/attn_encoder/"
        torch.save(model.model.encoder.state_dict(), MODEL_PATH + "weights.pt")
        with open(MODEL_PATH + "model_conf.pkl", "wb") as f:
            pickle.dump(model_conf, f)
        with open(MODEL_PATH + "lang.pkl", "wb") as f:
            joblib.dump(Lang, f)
        shutil.make_archive("./models/attn_encoder", "zip", "./models/attn_encoder")
        neptune_logger.experiment.log_artifact("./models/attn_encoder.zip")
