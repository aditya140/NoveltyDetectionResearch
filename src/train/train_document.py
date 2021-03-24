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
from src.datasets.document import *
from src.model.nli_models import *
from src.utils.trainer import *


class Train_document(Trainer):
    def __init__(
        self, args, dataset_conf, model_conf, hparams, model_type, sentence_field
    ):
        super(Train_document, self).__init__(
            args,
            model_conf,
            dataset_conf,
            hparams,
            log_neptune=True,
            **{
                "sentence_field": sentence_field,
                "neptune_project": DOC_NEPTUNE_PROJECT,
                "model_type": model_type,
            }
        )

    def load_dataset(self, dataset_conf, **kwargs):
        self.dataset = document_dataset(
            dataset_conf, sentence_field=kwargs["sentence_field"]
        )
        self.dataset.val_iter = self.dataset.test_iter

        self.label_size = len(self.dataset.labels())

        if self.log_neptune:
            neptune.append_tag([dataset_conf["dataset"], kwargs["model_type"]])
            neptune.log_text("class_labels", str(dict(self.dataset.labels())))
        if self.log_hyperdash:
            self.hd_exp.param("class_labels", str(dict(self.dataset.labels())))

    def load_model(self, model_conf, **kwargs):
        nli_model_data = load_encoder_data(self.args.load_nli)
        encoder = self.load_encoder(nli_model_data).encoder
        if model_conf["reset_enc"]:
            neptune.append_tag("encoder_reset")
            self.reset_encoder(encoder)
            print("Encoder Reset")
        model_conf["encoder_dim"] = nli_model_data["options"]["hidden_size"]

        if kwargs["model_type"] == "han":
            self.model = HAN_DOC_Classifier(model_conf, encoder)

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
        self.scheduler_has_args = False
        if hparams["optimizer"]["scheduler"] == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.5
            )

        elif hparams["optimizer"]["scheduler"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10
            )

        elif hparams["optimizer"]["scheduler"] == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=6, mode="max", threshold=0.01
            )
            self.scheduler_has_args = True

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

    @staticmethod
    def reset_encoder(enc):
        def weight_reset(m):
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        for name, module in enc.named_modules():
            if name not in ["embedding", "translate"]:
                module.apply(weight_reset)


if __name__ == "__main__":
    args = parse_document_clf_conf()
    (
        dataset_conf,
        optim_conf,
        model_type,
        model_conf,
        sentence_field,
    ) = get_document_clf_conf(args)
    trainer = Train_document(
        args, dataset_conf, model_conf, optim_conf, model_type, sentence_field
    )

    test_acc = trainer.fit(**{"batch_attr": {"model_inp": ["text"], "label": "label"}})
