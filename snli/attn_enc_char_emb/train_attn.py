import sys

sys.path.append(".")

import pytorch_lightning as pl
import torch.nn.functional as F
from datamodule import *
from pytorch_lightning.callbacks import LearningRateLogger
from snli.attn_enc_char_emb.attn_enc import *
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.metrics import Accuracy
import pickle
import os
import joblib
import shutil
import argparse
from lang import *
from snli.train_utils import (
    SNLI_model,
    SNLI_char_emb,
    snli_glove_data_module,
    snli_bert_data_module,
    SwitchOptim,
)
from utils.keys import NEPTUNE_API
from utils.helpers import seed_torch
from utils.save_models import save_model, save_model_neptune

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNLI Attention BiLSTM Training")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--save", action="store_true", help="Save model")
    args = parser.parse_args()

    seed_torch()
    print(args)

    data_module = snli_bert_data_module(128, char_emb=True)
    Lang = data_module.Lang
    embedding_matrix = None

    conf_kwargs = {
        "num_layers": 2,
        "dropout": 0.10018262692246818,
        "embedding_dim": 300,
        "hidden_size": 400,
        "attention_layer_param": 250,
        "bidirectional": True,
        "freeze_embedding": False,
        "activation": "tanh",
        "fcs": 1,
        "glove": False,
        "batch_size": 128,
        "max_len": 100,
    }

    hparams = {
        "optimizer_base": {
            "optim": "adamw",
            "lr": 0.0010039910781394373,
            "scheduler": "const",
        },
        "optimizer_tune": {
            "optim": "adam",
            "lr": 0.0010039910781394373,
            "weight_decay": 0.1,
            "scheduler": "lambda",
        },
        "switch_epoch": 6,
    }

    model_conf = Attn_Char_Encoder_conf(Lang, embedding_matrix, **conf_kwargs)

    model = SNLI_char_emb(Attn_encoder_snli, model_conf, hparams)

    EPOCHS = 6

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aparkhi/SNLI",
        experiment_name="Evaluation",
        tags=["Attention", "char_emb", args.tag],
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

    model_data = {"model": model.model.encoder, "model_conf": model_conf, "Lang": Lang}
    save_path = save_model("attn_char_encoder", expt_id, model_data)
    save_model_neptune(save_path, neptune_logger)
