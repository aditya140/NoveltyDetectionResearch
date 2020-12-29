import sys

sys.path.append(".")

import pytorch_lightning as pl
import torch.nn.functional as F
from utils import *
from datamodule import *
from pytorch_lightning.callbacks import LearningRateLogger
from snli.bert.bert import *
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.metrics import Accuracy
import pickle
import os
from joblib import Memory
import shutil
import argparse
from lang import *
from snli.train_utils import SNLI_bert, snli_glove_data_module, snli_bert_data_module
from transformers import DistilBertTokenizer, DistilBertModel
from utils.keys import NEPTUNE_API

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bert Training")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--save", action="store_true", help="Save model")
    args = parser.parse_args()


    encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    data_module = snli_bert_data_module(32,combine=True,tokenizer = "distilbert-base-uncased")
    Lang = data_module.Lang
    embedding_matrix = None


    hparams = {
        "optim": "adam",
        "lr": 3e-4,
        "scheduler": "lambda",
        "weight_decay":0
        }

    model_conf = Bert_Encoder_conf(**{"encoder":encoder})
    model = SNLI_bert(Bert_Encoder, model_conf, hparams)

    data_module.batch_size = 256

    EPOCHS = 10

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aditya140/SNLI",
        experiment_name="Evaluation",
        tags=["BERT", args.tag],
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

    if args.save:
        if not os.path.exists("models/bert_encoder"):
            os.makedirs("models/bert_encoder")
        BERT_PATH = "./models/bert_encoder/"
        torch.save(model.model.bert.state_dict(), BERT_PATH + "weights.pt")
        with open(BERT_PATH + "model_conf.pkl", "wb") as f:
            pickle.dump(model_conf, f)
        with open(BERT_PATH + "lang.pkl", "wb") as f:
            joblib.dump(Lang, f)
        shutil.make_archive("./models/bert_encoder", "zip", "./models/bert_encoder")
        neptune_logger.experiment.log_artifact("./models/bert_encoder.zip")
