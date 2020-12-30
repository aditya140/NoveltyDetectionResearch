import sys
import os
from joblib import Memory
import shutil
import joblib
import pickle
import copy
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.metrics import Accuracy

sys.path.append(".")

from lang import *
from novelty.cnn.cnn_model import *
from snli.bilstm.bilstm import *
from snli.attn_enc.attn_enc import *
from utils.load_models import load_bilstm_encoder, load_attn_encoder
from utils.save_models import save_model, save_model_neptune
from novelty.train_utils import *
from datamodule import *
from utils.keys import NEPTUNE_API


def test_fold(model, data_module, epochs):
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=EPOCHS,
        progress_bar_refresh_rate=10,
        profiler=False,
        auto_lr_find=False,
        row_log_interval=2,
        checkpoint_callback=False,
    )

    trainer.fit(model, data_module)
    test_res = trainer.test(model, datamodule=data_module)
    print("TEST_RESULTS = ", test_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty CNN Training")

    parser.add_argument("--webis", action="store_true", help="Webis dataset")
    parser.add_argument("--dlnd", action="store_true", help="DLND dataset")
    parser.add_argument("--apwsj", action="store_true", help="apwsj dataset")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    parser.add_argument("--epochs", type=int, help="Epochs")
    args = parser.parse_args()

    if args.encoder == "bilstm":
        model_id = "SNLI-13"
        encoder, Lang = load_bilstm_encoder(model_id)
    elif args.encoder == "attention":
        model_id = "SNLI-12"
        encoder, Lang = load_attn_encoder(model_id)

    if args.webis:
        data_module = webis_crossval_datamodule(Lang)
    elif args.dlnd:
        data_module = dlnd_crossval_datamodule(Lang)
    elif args.apwsj:
        data_module = apwsj_crossval_datamodule(Lang)

    params = {
        "num_filters": 60,
        "encoder_dim": encoder.conf.hidden_size,
        "dropout": 0.3,
        "expand features": True,
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

    init_state = copy.deepcopy(model.model.state_dict())
    EPOCHS = args.epochs

    for folds in range(10):
        data_module.set_fold(folds)
        model.model.load_state_dict(init_state)
        test_fold(model,data_module,EPOCHS)
