import sys

sys.path.append(".")

import pytorch_lightning as pl
import torch.nn.functional as F
from utils import *
from datamodule import *
from pytorch_lightning.callbacks import LearningRateLogger
from snli.struc_self_attn.struc_self_attn_enc import *
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.metrics import Accuracy
import pickle
import os
from joblib import Memory
import shutil
import argparse
from lang import *
from snli.train_utils import SNLI_model, snli_glove_data_module, snli_bert_data_module,SwitchOptim,SNLI_struc_attn_model
from keys import NEPTUNE_API

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
            "batch_size": 128,
            "max_len": 100,
            "embedding_dim": 300,
            "hidden_size" :300,
            "fcs" : 1,
            "r" : 30,
            "num_layers" : 2,
            "dropout" : 0.1,
            "opt_labels" : 3,
            "bidirectional" : True,
            "attention_layer_param" :100,
            "activation" : "tanh",
            "freeze_embedding" : False,
            "gated_embedding_dim" : 300,
            "gated" : False,
            "pool_strategy" : 'max',
            "C":0.0
        }

    hparams = {
        "optimizer_base":{
            "optim": "adamw",
            "lr": 0.0010039910781394373,
            "scheduler": "const"
            },
        "optimizer_tune":{
            "optim": "sgd",
            "lr": 3e-4,
            "momentum": 0,
            "weight_decay": 0,
            "scheduler": "const"
        },
        "switch_epoch":10,
    } 

    Lang = data_module.Lang

    model_conf = Struc_Attn_Encoder_conf(Lang, embedding_matrix,**conf_kwargs)
    # model = SNLI_model(Struc_Attn_encoder_snli, model_conf, hparams)
    model = SNLI_struc_attn_model(Struc_Attn_encoder_snli, model_conf, hparams)

    EPOCHS = 15

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aditya140/SNLI",
        experiment_name="Evaluation",
        tags=["Struc Self Attention"],
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
        gradient_clip_val=0.5,
    )
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)
        
    if args.save:
        if not os.path.exists("models/struc_attn_encoder"):
            os.makedirs("models/struc_attn_encoder")
        MODEL_PATH = "./models/struc_attn_encoder/"
        torch.save(model.model.encoder.state_dict(), MODEL_PATH + "weights.pt")
        with open(MODEL_PATH + "model_conf.pkl", "wb") as f:
            pickle.dump(model_conf, f)
        with open(MODEL_PATH + "lang.pkl", "wb") as f:
            joblib.dump(Lang, f)
        shutil.make_archive("./models/struc_attn_encoder", "zip", "./models/struc_attn_encoder")
        neptune_logger.experiment.log_artifact("./models/struc_attn_encoder.zip")


