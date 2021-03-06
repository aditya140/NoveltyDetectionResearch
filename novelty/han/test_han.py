import sys

sys.path.append(".")
import joblib
import pickle
import argparse
from lang import *
from novelty.han.han_novelty import *
from snli.bilstm.bilstm import *
from snli.attn_enc.attn_enc import *
from novelty.train_utils import *
import shutil
from utils.load_models import (
    load_bilstm_encoder,
    load_attn_encoder,
    load_han_clf_encoder,
    load_han_reg_encoder,
    reset_model,
)
from utils.helpers import seed_torch
import numpy as np
from novelty.train_utils import *
from datamodule import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.keys import NEPTUNE_API
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import random
import copy
import neptune


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty CNN Training")

    parser.add_argument("--webis", action="store_true", help="Webis dataset")
    parser.add_argument("--dlnd", action="store_true", help="DLND dataset")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    parser.add_argument(
        "--reset", action="store_true", help="Reset Weights", default=False
    )

    args = parser.parse_args()

    neptune.init(
        project_qualified_name="aparkhi/Novelty",
        api_token=NEPTUNE_API,
    )

    neptune.create_experiment()

    def train(model, data_module, optimizer, device):
        model.train()
        loss_values = []
        for batch in tqdm(data_module.train_dataloader()):
            x0, x1, y, id_ = batch
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
                x0, x1, y, id_ = batch
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

    seed_torch(140)
    neptune.log_text("Dataset", "Webis-CPC" if args.webis else "TAP-DLND")
    neptune.log_text("Encoder", args.encoder)

    if args.encoder == "reg":
        encoder, Lang = load_han_reg_encoder()
    elif args.encoder == "clf":
        encoder, Lang = load_han_clf_encoder()

    data_module = (
        WebisDataModule(batch_size=32, cross_val=True)
        if args.webis
        else DLNDDataModule(batch_size=32, cross_val=True)
    )

    neptune.log_metric("Batch_size", 32)

    print("Started data prep")
    data_module.prepare_data(Lang, 100)
    print("Data Prepared")

    params = {}

    hparams = {
        "optimizer_base":{
            "optim": "adamw",
            "lr": 0.0010039910781394373,
            "scheduler": "const"
            },
        "optimizer_tune":{
            "optim": "adam",
            "lr": 0.00010039910781394373,
            "weight_decay": 0.1,
            "scheduler": "lambda"
        },
        "switch_epoch":3,
    }

    neptune.log_text("hparams", hparams.__str__())
    neptune.log_text("params", params.__str__())

    model_conf = HAN_Novelty_conf(encoder, **pparams)
    model = Novelty_model(HAN_Novelty, model_conf, hparams)

    if args.reset:
        print("Reinitializing weights")
        model.model = reset_model(model.model)

    optimizer = optim.AdamW(model.parameters(), lr=hparams["lr"])
    init_state = copy.deepcopy(model.state_dict())
    init_state_opt = copy.deepcopy(optimizer.state_dict())

    overall_loss, overall_acc, overall_prec, overall_recal, overall_f1 = 0, 0, 0, 0, 0
    for folds in range(10):
        print("--" * 10)
        print(f"Fold {folds}:")
        if args.encoder == "reg":
            encoder, Lang = load_han_reg_encoder()
        elif args.encoder == "clf":
            encoder, Lang = load_han_clf_encoder()

        data_module.set_fold(folds)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_conf = HAN_Novelty_conf(encoder, **hparams)
        model = Novelty_model(HAN_Novelty, model_conf, params)

        optimizer = optim.AdamW(model.parameters(), lr=hparams["lr"])
        model.load_state_dict(init_state)
        optimizer.load_state_dict(init_state_opt)
        model.to(device)

        EPOCHS = 5
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
