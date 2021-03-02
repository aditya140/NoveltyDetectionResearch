import sys

sys.path.append(".")
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import datetime
import time
import shutil
import neptune
from millify import millify


from src.defaults import *
from src.model.novelty_models import *
from src.datasets.novelty import *
from src.model.nli_models import *
from src.utils.trainer import *


class Train_novelty(Trainer):
    def __init__(
        self, args, dataset_conf, model_conf, hparams, model_type, sentence_field
    ):
        super(Train_novelty, self).__init__(
            args,
            model_conf,
            dataset_conf,
            hparams,
            log_neptune=True,
            **{
                "sentence_field": sentence_field,
                "neptune_project": NOVELTY_NEPTUNE_PROJECT,
                "model_type": model_type,
            }
        )

    def load_dataset(self, dataset_conf, **kwargs):
        self.dataset = novelty_dataset(
            dataset_conf, sentence_field=kwargs["sentence_field"]
        )

        lable_dict = self.dataset.labels()

        if self.log_neptune:
            neptune.append_tag([dataset_conf["dataset"], kwargs["model_type"]])
            neptune.log_text("class_labels", str(dict(self.dataset.labels())))
        if self.log_hyperdash:
            self.hd_exp.param("class_labels", str(dict(self.dataset.labels())))

    def load_model(self, model_conf, **kwargs):
        nli_model_data = load_encoder_data(self.args.load_nli)
        encoder = self.load_encoder(nli_model_data).encoder
        model_conf["encoder_dim"] = nli_model_data["options"]["hidden_size"]

        if kwargs["model_type"] == "dan":
            self.model = DAN(model_conf, encoder)
        if kwargs["model_type"] == "adin":
            self.model = ADIN(model_conf, encoder)
        if kwargs["model_type"] == "han":
            self.model = HAN(model_conf, encoder)
        if kwargs["model_type"] == "rdv_cnn":
            self.model = RDV_CNN(model_conf, encoder)
        if kwargs["model_type"] == "diin":
            self.model = DIIN(model_conf, encoder)
        if kwargs["model_type"] == "mwan":
            self.model = MwAN(model_conf, encoder)
        if kwargs["model_type"] == "struc":
            self.model = StrucSelfAttn(model_conf, encoder)

        self.model.to(self.device)

    def set_optimizers(self, hparams, **kwargs):
        self.criterion = nn.CrossEntropyLoss(reduction=hparams["loss_agg"])
        if hparams["optimizer"]["optim"] == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=hparams["optimizer"]["lr"]
            )

        elif hparams["optimizer"]["optim"] == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=hparams["optimizer"]["lr"]
            )

        elif hparams["optimizer"]["optim"] == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=hparams["optimizer"]["lr"],
                momentum=0.9,
            )

        elif hparams["optimizer"]["optim"] == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=hparams["optimizer"]["lr"],
            )

        else:
            raise ValueError(
                "Wrong optimizer type, select from adam, adamw, sgd, adadelta"
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


if __name__ == "__main__":
    args = parse_novelty_conf()
    dataset_conf, optim_conf, model_type, model_conf, sentence_field = get_novelty_conf(
        args
    )
    trainer = Train_novelty(
        args, dataset_conf, model_conf, optim_conf, model_type, sentence_field
    )
    if args.folds:
        test_acc = trainer.test_folds(
            **{
                "model_type": model_type,
                "batch_attr": {"model_inp": ["source", "target"], "label": "label"},
            }
        )
    else:
        test_acc = trainer.fit(
            **{"batch_attr": {"model_inp": ["source", "target"], "label": "label"}}
        )
