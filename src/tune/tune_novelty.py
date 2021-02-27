import sys

sys.path.append(".")
import warnings

warnings.filterwarnings("ignore")

import datetime
import time
import shutil
import neptune
from millify import millify
import pprint
import copy
import os
from functools import partial
from tabulate import tabulate
from hyperdash import Experiment

from src.defaults import *
from src.model.novelty_models import *
from src.datasets.novelty import *
from src.model.nli_models import *
from src.utils.tuner import *


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import optuna
from optuna.samplers import TPESampler


class Tune_novelty(Tuner):
    def __init__(
        self, args, dataset_conf, model_conf, hparams, model_type, sentence_field
    ):
        self.args = args
        super().__init__(
            dataset_conf,
            model_conf,
            hparams,
            **{"sentence_field": sentence_field, "model_type": model_type},
        )

    def load_dataset(self):
        if dataset_conf["dataset"] == "dlnd":
            dataset = dlnd(
                self.dataset_conf, sentence_field=self.init_kwargs["sentence_field"]
            )
        return dataset

    def load_model(
        self,
    ):
        nli_model_data = load_encoder_data(self.args.load_nli)
        encoder = self.load_encoder(nli_model_data).encoder
        model_conf["encoder_dim"] = nli_model_data["options"]["hidden_size"]

        if self.init_kwargs["model_type"] == "dan":
            model = DAN
        if self.init_kwargs["model_type"] == "adin":
            model = ADIN
        if self.init_kwargs["model_type"] == "han":
            model = HAN
        if self.init_kwargs["model_type"] == "rdv_cnn":
            model = RDV_CNN
        if self.init_kwargs["model_type"] == "diin":
            model = DIIN
        if self.init_kwargs["model_type"] == "mwan":
            model = MwAN
        if self.init_kwargs["model_type"] == "struc":
            model = StrucSelfAttn

        return model, {"encoder": encoder}

    @staticmethod
    def load_encoder(enc_data):
        if enc_data["options"].get("attention_layer_param", 0) == 0:
            enc_data["options"]["use_glove"] = False
            model = bilstm_snli(enc_data["options"])
        elif enc_data["options"].get("r", 0) == 0:
            enc_data["options"]["use_glove"] = False
            model = attn_bilstm_snli(enc_data["options"])
        else:
            enc_data["options"]["use_glove"] = False
            model = struc_attn_snli(enc_data["options"])
        model.load_state_dict(enc_data["model_dict"])
        return model


def get_optimizer_criterion(net, hparams):
    criterion = nn.CrossEntropyLoss()
    if hparams["optimizer"]["optim"] == "adam":
        optimizer = optim.Adam(net.parameters(), lr=hparams["optimizer"]["lr"])

    elif hparams["optimizer"]["optim"] == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=hparams["optimizer"]["lr"])

    elif hparams["optimizer"]["optim"] == "sgd":
        optimizer = optim.SGD(
            net.parameters(),
            lr=hparams["optimizer"]["lr"],
            momentum=0.9,
        )

    elif hparams["optimizer"]["optim"] == "adadelta":
        optimizer = optim.Adadelta(
            net.parameters(),
            lr=hparams["optimizer"]["lr"],
        )

    else:
        raise ValueError("Wrong optimizer type, select from adam, adamw, sgd, adadelta")
    return optimizer, criterion


def objective(
    trial, model, model_kwargs, model_type, dataset, model_config, hparams, epochs
):
    model_kwargs = copy.deepcopy(model_kwargs)
    model_config = model_conf_tuning(trial, model_config, model_type)
    net = model(model_config, **model_kwargs)
    net.to(device)
    optimizer, criterion = get_optimizer_criterion(net, hparams)

    best_val_acc = 0

    for epoch in range(epochs):
        train_dl, val_dl, test_dl = (
            dataset.train_iter,
            dataset.val_iter,
            dataset.test_iter,
        )
        train_loss, train_acc = train_model(
            net,
            optimizer,
            criterion,
            train_dl,
            **{
                "batch_attr": {"model_inp": ["source", "target"], "label": "label"},
            },
        )
        val_loss, val_acc = validate_model(
            net,
            optimizer,
            criterion,
            val_dl,
            **{
                "batch_attr": {"model_inp": ["source", "target"], "label": "label"},
            },
        )

        trial.report(val_acc, epoch)
        best_val_acc = max(best_val_acc,val_acc)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return best_val_acc


def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


if __name__ == "__main__":

    exp = Experiment("tuning", api_key_getter=get_hyperdash_api)

    args = parse_novelty_tune_conf()
    (
        dataset_conf,
        optim_conf,
        model_type,
        model_conf,
        sentence_field,
    ) = get_tuning_novelty_conf(args)

    tuner = Tune_novelty(
        args, dataset_conf, model_conf, optim_conf, model_type, sentence_field
    )
    print(args)

    dataset = tuner.load_dataset()
    model, model_kwargs = tuner.load_model()
    device = tuner.device

    partial_objective = partial(
        objective,
        model=model,
        model_kwargs=model_kwargs,
        model_type=model_type,
        dataset=dataset,
        model_config=model_conf,
        hparams=optim_conf,
        epochs=args.epochs,
    )

    if args.sampler == "tpe":
        sampler = TPESampler()
    if args.sampler == "grid":
        search_space = model_search_space(model_type)
        sampler = optuna.samplers.GridSampler(search_space)

    study = optuna.create_study(
        direction="maximize", study_name="novelty_tuner", sampler=sampler
    )

    study.optimize(
        partial_objective, n_trials=args.num_trials, callbacks=[print_best_callback]
    )
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    exp.log("Study statistics: ")
    exp.log(f"  Number of finished trials: {len(study.trials)}")
    exp.log(f"  Number of pruned trials: {len(pruned_trials)}")
    exp.log(f"  Number of complete trials: {len(complete_trials)}")

    print("Best trial:")
    exp.log("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    exp.log(f" Value: {trial.value}")

    print("  Params: ")
    exp.log("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        exp.log("    {}: {}".format(key, value))

    print("Study Dataframe: ")
    print(tabulate(study.trials_dataframe(), headers="keys", tablefmt="psql"))
    exp.log(tabulate(study.trials_dataframe(), headers="keys", tablefmt="psql"))
    exp.end()