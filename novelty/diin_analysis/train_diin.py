import sys

sys.path.append(".")
import joblib
import pickle
import argparse
from lang import *
from novelty.cnn.cnn_model import *
from novelty.diin.diin import *
from utils import seed_torch
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
from novelty.train_utils import *
from datamodule import *
import os
from utils.keys import NEPTUNE_API




if __name__ == "__main__":
    seed_torch()
    parser = argparse.ArgumentParser(description="Novelty CNN Training")

    parser.add_argument(
        "--webis", action="store_true", help="Webis dataset", default=False
    )
    parser.add_argument(
        "--dlnd", action="store_true", help="DLND dataset", default=False
    )
    parser.add_argument("--save", action="store_true", help="Save model")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    parser.add_argument("--batch_size", type=int, help="batch size", default=25)

    args = parser.parse_args()

    if args.encoder == "bilstm":
        encoder, Lang = load_bilstm_encoder()
    elif args.encoder == "attention":
        encoder, Lang = load_attn_encoder()
        
    data_module = webis_data_module(Lang) if args.webis else dlnd_data_module(Lang)
    data_module.batch_size = args.batch_size

    params = {
        "optim": "adamw",
        "lr": 0.00060869262115700171,
        "weight_decay":0.01,
        "scheduler": "constant",
        "encoder_type": args.encoder,
        "batch_size": args.batch_size,
    }

    params = {
        "optim": "adamw",
        "lr": 0.00060869262115700171,
        "weight_decay":0.01,
        "scheduler": "constant",
        "encoder_type": args.encoder,
        "batch_size": args.batch_size,
    }

    # params = {
    #     "optim": "sgd",
    #     "lr": 1,
    #     "momentum":0,
    #     "weight_decay":0,
    #     "scheduler": "constant",
    #     "encoder_type": args.encoder,
    #     "batch_size": args.batch_size,
    # }

    model_conf = DIIN_conf(100, encoder, **params)
    model = Novelty_CNN_model(DIIN, model_conf, params)

    EPOCHS = 20

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aditya140/Novelty",
        experiment_name="Evaluation",  # Optional,
        tags=[("Webis" if args.webis else "DLND"), "test", "DIIN"],
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
        MODEL_PATH = "./models/diin_novelty/"
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        torch.save(model.model.state_dict(), MODEL_PATH + "weights.pt")
        with open(MODEL_PATH + "model_conf.pkl", "wb") as f:
            pickle.dump(model_conf, f)
        with open(MODEL_PATH + "lang.pkl", "wb") as f:
            pickle.dump(Lang, f)
        shutil.make_archive("./models/diin_novelty", "zip", "./models/diin_novelty")
        neptune_logger.experiment.log_artifact("./models/diin_novelty.zip")
