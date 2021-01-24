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
import neptune

sys.path.append(".")

from lang import *
from novelty.han_cnn.han_cnn import *
from snli.bilstm.bilstm import *
from snli.attn_enc.attn_enc import *
from utils.load_models import (
    load_han_attn_encoder,
    load_han_bilstm_encoder,
    reset_model,
)
from utils.save_models import save_model, save_model_neptune
from novelty.train_utils import *
from datamodule import *
from utils.keys import NEPTUNE_API
from utils.helpers import seed_torch


def test_fold(model, data_module, epochs):
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epochs,
        progress_bar_refresh_rate=10,
        profiler=False,
        auto_lr_find=False,
        row_log_interval=2,
        checkpoint_callback=False,
    )
    trainer.fit(model, data_module)
    test_res = trainer.test(model, datamodule=data_module)[0]
    test_loss = test_res["test_loss"]
    test_acc = test_res["test_acc"]
    test_f1 = test_res["test_f1"]
    test_recall = test_res["test_recall"]
    test_prec = test_res["test_prec"]
    with open("./analysis/tempids.txt", "r") as f:
        error_ids = [int(i) for i in f.read().split(",")]

    return test_loss, test_acc, test_f1, test_recall, test_prec, error_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty DAN Training")

    parser.add_argument("--webis", action="store_true", help="Webis dataset")
    parser.add_argument("--dlnd", action="store_true", help="DLND dataset")
    parser.add_argument("--apwsj", action="store_true", help="apwsj dataset")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    parser.add_argument("--epochs", type=int, help="Epochs")
    parser.add_argument("--reset", action="store_true", help="reset weights")
    parser.add_argument(
        "--use_nltk", action="store_true", help="Dataset imdb", default=False
    )

    parser.add_argument(
        "--analysis", action="store_true", help="Anaysis mode", default=False
    )
    args = parser.parse_args()

    use_nltk = args.use_nltk

    if args.encoder == "bilstm":
        model_id = "DOC-5"
        encoder, Lang = load_han_bilstm_encoder(model_id)
    elif args.encoder == "attention":
        # model_id = "DOC-2"
        model_id = "DOC-13"
        encoder, Lang = load_han_attn_encoder(model_id)

    if args.webis:
        data_module = webis_crossval_datamodule(Lang, use_nltk)
        data_module.batch_size = 12
    elif args.dlnd:
        data_module = dlnd_crossval_datamodule(Lang, use_nltk)
        data_module.batch_size = 12
    elif args.apwsj:
        data_module = apwsj_crossval_datamodule(Lang, use_nltk)
        data_module.batch_size = 12

    neptune.init(
        project_qualified_name="aparkhi/Novelty",
        api_token=NEPTUNE_API,
    )

    neptune.create_experiment(
        tags=["10-fold", "HAN-CNN", ("no_pretraining" if args.reset else "pretrained")]
    )
    seed_torch(140)

    neptune.append_tag(("Webis" if args.webis else ("DLND" if args.dlnd else "APWSJ")))

    neptune.log_text(
        "Dataset", ("Webis" if args.webis else ("DLND" if args.dlnd else "APWSJ"))
    )
    neptune.log_text("Encoder", args.encoder)
    neptune.log_text("Use NLTK", str(use_nltk))

    params = {
        "optim": "adamw",
        "weight_decay": 0.1,
        "lr": 0.00010869262115700171,
        "scheduler": "lambda",
        "activation": "tanh",
        "analysis": args.analysis,
        "analysisFile": "temp.csv",
    }
    neptune.log_text("params", params.__str__())
    neptune.log_text("epochs", str(args.epochs))

    model_conf = HAN_CNN_conf(100, encoder, **params)
    model = Novelty_CNN_model(HAN_CNN, model_conf, params)

    if args.reset:
        print("Reinitializing weights")
        model.model = reset_model(model.model)

    init_state = copy.deepcopy(model.model.state_dict())
    EPOCHS = args.epochs

    overall_loss, overall_acc, overall_prec, overall_recal, overall_f1 = 0, 0, 0, 0, 0
    all_error_ids = []
    for folds in range(10):
        data_module.set_fold(folds)
        model.model.load_state_dict(init_state)
        test_loss, test_acc, test_f1, test_recall, test_prec, error_ids = test_fold(
            model, data_module, EPOCHS
        )
        neptune.log_metric("test_loss", test_loss)
        neptune.log_metric("test_acc", test_acc)
        neptune.log_metric("test_prec", test_prec)
        neptune.log_metric("test_recall", test_recall)
        neptune.log_metric("test_f1", test_f1)
        overall_loss += test_loss
        overall_acc += test_acc
        overall_prec += test_prec
        overall_recal += test_recall
        overall_f1 += test_f1
        if error_ids != None:
            all_error_ids += error_ids

    overall_loss, overall_acc, overall_prec, overall_recal, overall_f1 = (
        overall_loss / 10,
        overall_acc / 10,
        overall_prec / 10,
        overall_recal / 10,
        overall_f1 / 10,
    )

    print(
        "Final Accuracy: {overall_acc}, Precsion: {overall_prec}, Recall: {overall_recal}, F1 Score: {overall_f1}"
    )
    neptune.log_metric("final_loss", overall_loss)
    neptune.log_metric("final_acc", overall_acc)
    neptune.log_metric("final_prec", overall_prec)
    neptune.log_metric("final_recall", overall_recal)
    neptune.log_metric("final_f1", overall_f1)
    neptune.stop()

    if all_error_ids != []:
        with open("analysis/han_cnn.txt", "w") as f:
            f.write(",".join([str(i) for i in all_error_ids]))