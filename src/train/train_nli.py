import sys, os

sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
import shutil
import neptune

from src.defaults import *
from src.model.nli_models import *
from src.datasets.nli import *
from src.utils.trainer import *


class Train_nli(Trainer):
    def __init__(
        self,
        args,
        dataset_conf,
        model_conf,
        hparams,
        model_type,
    ):
        super(Train_nli, self).__init__(
            args,
            model_conf,
            dataset_conf,
            hparams,
            log_neptune=True,
            **{
                "neptune_project": NLI_NEPTUNE_PROJECT,
                "model_type": model_type,
            }
        )

    def load_dataset(self, dataset_conf, **kwargs):
        if dataset_conf["dataset"] == "snli":
            self.dataset = snli_module(dataset_conf)
        elif dataset_conf["dataset"] == "mnli":
            self.dataset = mnli_module(dataset_conf)

        self.dataset.prepare_data()
        self.dataset = self.dataset.data
        self.label_size = len(self.dataset.labels())
        if self.log_neptune:
            neptune.append_tag([dataset_conf["dataset"], kwargs["model_type"]])

    def load_model(self, model_conf, **kwargs):
        model_conf["vocab_size"] = self.dataset.vocab_size()
        model_conf["padding_idx"] = self.dataset.padding_idx()

        if args.use_char_emb:
            model_conf["char_vocab_size"] = self.dataset.char_vocab_size()

        if model_type == "attention":
            self.model = attn_bilstm_snli(model_conf)
        elif model_type == "bilstm":
            self.model = bilstm_snli(model_conf)
        elif model_type == "struc_attn":
            self.model = struc_attn_snli(model_conf)
        elif model_type == "mwan":
            self.model = mwan_snli(model_conf)
        elif model_type == 'bert':
            self.model = bert_snli(model_conf)

        self.model.to(self.device)
        model_size = self.count_parameters(self.model)
        print(" [*] Model size : {}".format(model_size))

        self.logger.info(" [*] Model size : {}".format(model_size))
        if self.log_neptune:
            neptune.log_text("Model size", str(model_size))

    def set_optimizers(self, hparams, **kwargs):
        self.criterion = nn.CrossEntropyLoss(reduction=hparams["loss_agg"])
        if hparams["optimizer"]["optim"] == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=hparams["optimizer"]["lr"]
            )

        if hparams["optimizer"]["optim"] == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=hparams["optimizer"]["lr"]
            )

        if hparams["optimizer"]["optim"] == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=hparams["optimizer"]["lr"],
                momentum=0.9,
            )

        self.best_val_acc = None

    def set_schedulers(self, hparams, **kwargs):
        if hparams["optimizer"]["scheduler"] == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.5
            )

        elif hparams["optimizer"]["scheduler"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10
            )

        else:
            self.scheduler = None

    def save_lang(self):
        text_field, char_field = get_vocabs(self.dataset)
        save_field(
            os.path.join(
                self.args.results_dir,
                self.exp_id,
                "text_field",
            ),
            text_field,
        )
        if char_field != None:
            save_field(
                os.path.join(
                    self.args.results_dir,
                    self.exp_id,
                    "char_field",
                ),
                char_field,
            )

    def save_to_neptune(self):
        shutil.make_archive(
            os.path.join(
                self.args.results_dir,
                self.exp_id,
            ),
            "zip",
            os.path.join(
                self.args.results_dir,
                self.exp_id,
            ),
        )
        neptune.log_artifact(
            os.path.join(
                self.args.results_dir,
                self.exp_id + ".zip",
            )
        )

    def save(self):
        self.save_lang()
        self.save_to_neptune()


if __name__ == "__main__":
    args = parse_nli_conf()
    dataset_conf, optim_conf, model_type, model_conf = get_nli_conf(args)
    trainer = Train_nli(
        args,
        dataset_conf,
        model_conf,
        optim_conf,
        model_type,
    )
    trainer.fit(
        **{"batch_attr": {"model_inp": ["premise", "hypothesis"], "label": "label"}}
    )
