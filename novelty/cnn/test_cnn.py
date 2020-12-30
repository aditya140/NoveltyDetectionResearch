import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import random
import copy
import neptune
import joblib
from joblib import memory
import pickle
import argparse
import shutil
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


sys.path.append(".")
from lang import *
from novelty.cnn.cnn_model import *
from snli.bilstm.bilstm import *
from snli.attn_enc.attn_enc import *
from utils.load_models import load_bilstm_encoder, load_attn_encoder
from novelty.train_utils import *
from datamodule import *
from utils.keys import NEPTUNE_API
from utils.helpers import seed_torch


@memory.cache
def webis_crossval_datamodule(Lang):
    data_module = WebisDataModule(batch_size=32, cross_val=True)
    print("Started data prep")
    data_module.prepare_data(Lang, 100)
    print("Data Prepared")
    return data_module


@memory.cache
def dlnd_crossval_datamodule(Lang):
    data_module = DLNDDataModule(batch_size=32, cross_val=True)
    print("Started data prep")
    data_module.prepare_data(Lang, 100)
    print("Data Prepared")
    return data_module


@memory.cache
def apwsj_crossval_datamodule(Lang):
    data_module = APWSJDataModule(batch_size=32, cross_val=True)
    print("Started data prep")
    data_module.prepare_data(Lang, 100)
    print("Data Prepared")
    return data_module


def train(model, data_module, optimizer, device):
    model.train()
    loss_values = []
    for batch in tqdm(data_module.train_dataloader()):
        x0, x1, y = batch
        model.zero_grad()
        opt = model(x0.to(device), x1.to(device)).squeeze(1)
        loss = F.cross_entropy(opt, y.to(device))
        loss.backward()
        loss_values.append(loss.cpu().item())
        optimizer.step()
    return np.mean(loss_values)


def evaluate(model, data_module, device):
    loss_values = []
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_module.test_dataloader()):
            x0, x1, y = batch
            model.zero_grad()
            opt = model(x0.to(device), x1.to(device)).squeeze(1)
            loss = F.cross_entropy(opt, y.to(device))
            pred = F.softmax(opt)

            loss_values.append(loss.cpu().item())
            precision, recall, f1, _ = precision_recall_fscore_support(
                y.cpu().numpy(), pred.argmax(1).cpu().numpy(), average="macro"
            )
            accuracy_values.append(
                accuracy_score(y.cpu().numpy(), pred.argmax(1).cpu().numpy())
            )
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

    return (
        np.mean(loss_values),
        np.mean(accuracy_values),
        np.mean(precision_values),
        np.mean(recall_values),
        np.mean(f1_values),
    )


def main(args):

    neptune.init(
        project_qualified_name="aparkhi/Novelty",
        api_token=NEPTUNE_API,
    )

    neptune.create_experiment(tags=["10-fold"])

    seed_torch(140)
    neptune.log_text(
        "Dataset", ("Webis" if args.webis else ("DLND" if args.dlnd else "APWSJ"))
    )
    neptune.log_text("Encoder", args.encoder)

    if args.encoder == "bilstm":
        model_id = "SNLI-13"
        encoder, Lang = load_bilstm_encoder(model_id)
    elif args.encoder == "attention":
        model_id = "SNLI-12"
        encoder, Lang = load_attn_encoder(model_id)

    neptune.append_tag(args.encoder)
    neptune.append_tag(("Webis" if args.webis else ("DLND" if args.dlnd else "APWSJ")))

    if args.webis:
        data_module = webis_crossval_datamodule(Lang)
    elif args.dlnd:
        data_module = dlnd_crossval_datamodule(Lang)
    elif args.apwsj:
        data_module = apwsj_crossval_datamodule(Lang)

    params = {
        "num_filters": 100,
        "encoder_dim": encoder.conf.hidden_size,
        "dropout": 0.3,
        "expand features": True,
        "filter_sizes": [4, 6, 9],
        "freeze_embedding": True,
        "activation": "tanh",
    }

    hparams = {
        "optim": "adamw",
        "weight_decay": 0.1,
        "lr": 0.00010869262115700171,
        "scheduler": "lambda",
    }

    neptune.log_text("hparams", hparams.__str__())
    neptune.log_text("params", params.__str__())
    neptune.log_text("epochs", str(args.epochs))

    model_conf = Novelty_CNN_conf(100, encoder, **params)
    model = DeepNoveltyCNN(model_conf)
    optimizer = optim.AdamW(model.parameters(), lr=hparams["lr"])
    init_state = copy.deepcopy(model.state_dict())
    init_state_opt = copy.deepcopy(optimizer.state_dict())

    overall_loss, overall_acc, overall_prec, overall_recal, overall_f1 = 0, 0, 0, 0, 0
    for folds in range(10):
        print("--" * 10)
        print(f"Fold {folds}:")

        if args.encoder == "bilstm":
            model_id = "SNLI-13"
            encoder, Lang = load_bilstm_encoder(model_id)
        elif args.encoder == "attention":
            model_id = "SNLI-12"
            encoder, Lang = load_attn_encoder(model_id)

        data_module.set_fold(folds)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_conf = Novelty_CNN_conf(100, encoder, **params)
        model = DeepNoveltyCNN(model_conf)
        optimizer = optim.AdamW(model.parameters(), lr=hparams["lr"])
        model.load_state_dict(init_state)
        optimizer.load_state_dict(init_state_opt)
        model.to(device)

        EPOCHS = args.epochs
        for ep in range(EPOCHS):
            train_loss = train(model, data_module, optimizer, device)
            print(f"\tTraining Loss => epoch {ep}: {train_loss}")

        test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate(
            model, data_module, device
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
        print(f"\tTest stats:")
        print(
            f"\t\t Loss : {test_loss}, Accuracy: {test_acc}, Precsion: {test_prec}, Recall: {test_recall}, F1 Score: {test_f1}"
        )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty CNN Training")
    parser.add_argument("--webis", action="store_true", help="Webis dataset")
    parser.add_argument("--dlnd", action="store_true", help="DLND dataset")
    parser.add_argument("--apwsj", action="store_true", help="apwsj dataset")
    parser.add_argument("--epochs", type=int, help="Epochs")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    args = parser.parse_args()

    main(args)
