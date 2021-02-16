import sys

sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import datetime
import time
import shutil
import neptune

from src.defaults import *
from src.model.novelty_models import *
from src.datasets.novelty import *
from src.model.nli_models import *


class Train:
    def __init__(
        self,
        args,
        dataset_conf,
        model_conf,
        hparams,
        model_type,
        sentence_field,
    ):

        ### Basic setup
        ###################################################

        print("program execution start: {}".format(datetime.datetime.now()))
        neptune.init(
            project_qualified_name=NOVELTY_NEPTUNE_PROJECT,
            api_token=NEPTUNE_API,
        )

        self.exp = neptune.create_experiment()
        self.exp_id = self.exp.id

        self.args = args
        self.device = get_device(0)
        self.logger = get_logger(self.args, "train", self.exp_id)

        self.logger.info("Dataset Conf: {}".format(dataset_conf))
        neptune.log_text("Dataset Conf", str(dataset_conf))

        self.logger.info("Model Conf: {}".format(model_conf))
        neptune.log_text("Model Conf", str(model_conf))
        self.model_conf = model_conf
        self.dataset_conf = dataset_conf

        self.logger.info("Hparams Conf: {}".format(hparams))
        neptune.log_text("Hparams", str(hparams))

        ### Load Dataset
        ###################################################

        if dataset_conf["dataset"] == "dlnd":
            self.dataset = dlnd(dataset_conf, sentence_field=sentence_field)

        neptune.append_tag([dataset_conf["dataset"], model_type])

        ### Load Model
        ###################################################

        nli_model_data = load_encoder_data(args.load_nli)

        encoder = self.load_encoder(nli_model_data).encoder

        model_conf["encoder_dim"] = nli_model_data["options"]["hidden_size"]

        if model_type == "dan":
            self.model = DAN(model_conf, encoder)
        if model_type == 'adin':
            self.model = ADIN(model_conf,encoder)
        if model_type == 'han':
            self.model = HAN(model_conf,encoder)

        self.model.to(self.device)
        model_size = self.count_parameters()
        print(" [*] Model size : {}".format(model_size))

        self.logger.info(" [*] Model size : {}".format(model_size))
        neptune.log_text("Model size", str(model_size))

        ### Optimizer setup
        ###################################################

        self.criterion = nn.CrossEntropyLoss(reduction=hparams["loss_agg"])
        self.softmax = nn.Softmax(dim=1)

        if hparams["optimizer"]["optim"] == "adam":
            self.opt = optim.Adam(
                self.model.parameters(), lr=hparams["optimizer"]["lr"]
            )

        if hparams["optimizer"]["optim"] == "adamw":
            self.opt = optim.AdamW(
                self.model.parameters(), lr=hparams["optimizer"]["lr"]
            )

        if hparams["optimizer"]["optim"] == "sgd":
            self.opt = optim.SGD(
                self.model.parameters(),
                lr=hparams["optimizer"]["lr"],
                momentum=0.9,
            )

        self.best_val_acc = None

        if hparams["optimizer"]["scheduler"] == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=5, gamma=0.5)

        elif hparams["optimizer"]["scheduler"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=10)

        else:
            self.scheduler = None

        print("resource preparation done: {}".format(datetime.datetime.now()))

    def result_checkpoint(self, epoch, train_loss, val_loss, train_acc, val_acc, took):
        if self.best_val_acc is None or val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(
                {
                    "accuracy": self.best_val_acc,
                    "options": self.model_conf,
                    "model_dict": self.model.state_dict(),
                    "dataset_conf": self.dataset_conf,
                },
                "{}/{}/model.pt".format(
                    self.args.results_dir,
                    self.exp_id,
                ),
            )

        self.logger.info(
            "| Epoch {:3d} | train loss {:5.2f} | train acc {:5.2f} | val loss {:5.2f} | val acc {:5.2f} | time: {:5.2f}s |".format(
                epoch, train_loss, train_acc, val_loss, val_acc, took
            )
        )

    def train(self):
        self.model.train()
        self.dataset.train_iter.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0
        for batch_idx, batch in enumerate(self.dataset.train_iter):

            self.opt.zero_grad()
            answer = self.model(batch.source, batch.target)
            loss = self.criterion(answer, batch.label)
            n_correct += (
                (
                    torch.max(self.softmax(answer), 1)[1].view(batch.label.size())
                    == batch.label
                )
                .sum()
                .item()
            )
            n_total += batch.batch_size
            n_loss += loss.item()
            neptune.log_metric("Train Loss", loss.item())
            loss.backward()
            self.opt.step()
        train_loss = n_loss / n_total
        train_acc = 100.0 * n_correct / n_total
        neptune.log_metric("Train Avg Loss", train_loss)
        neptune.log_metric("Train accuracy", train_acc)
        return train_loss, train_acc

    def validate(self):
        self.model.eval()
        self.dataset.val_iter.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataset.val_iter):

                answer = self.model(batch.source, batch.target)
                loss = self.criterion(answer, batch.label)
                n_correct += (
                    (
                        torch.max(self.softmax(answer), 1)[1].view(batch.label.size())
                        == batch.label
                    )
                    .sum()
                    .item()
                )
                n_total += batch.batch_size
                n_loss += loss.item()
                neptune.log_metric("Val Loss", loss.item())
            val_loss = n_loss / n_total
            val_acc = 100.0 * n_correct / n_total
            neptune.log_metric("Val Avg Loss", val_loss)
            neptune.log_metric("Val Accuracy", val_acc)
            return val_loss, val_acc

    def test(self):
        PATH = "{}/{}/model.pt".format(
            self.args.results_dir,
            self.exp_id,
        )
        model_data = torch.load(PATH)
        self.model.load_state_dict(model_data["model_dict"])
        self.model.eval()
        self.dataset.test_iter.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataset.test_iter):

                answer = self.model(batch.source, batch.target)
                loss = self.criterion(answer, batch.label)
                n_correct += (
                    (
                        torch.max(self.softmax(answer), 1)[1].view(batch.label.size())
                        == batch.label
                    )
                    .sum()
                    .item()
                )
                n_total += batch.batch_size
                n_loss += loss.item()
            test_loss = n_loss / n_total
            test_acc = 100.0 * n_correct / n_total
            neptune.log_metric("Test Avg Loss", test_loss)
            neptune.log_metric("Test Accuracy", test_acc)
            return test_loss, test_acc

    def execute(self):
        print(" [*] Training starts!")
        print("-" * 99)
        for epoch in range(1, self.args.epochs + 1):
            start = time.time()

            train_loss, train_acc = self.train()
            val_loss, val_acc = self.validate()
            if self.scheduler != None:
                self.scheduler.step()

            took = time.time() - start
            self.result_checkpoint(
                epoch, train_loss, val_loss, train_acc, val_acc, took
            )

            print(
                "| Epoch {:3d} | train loss {:5.2f} | train acc {:5.2f} | val loss {:5.2f} | val acc {:5.2f} | time: {:5.2f}s |".format(
                    epoch, train_loss, train_acc, val_loss, val_acc, took
                )
            )
        test_loss, test_acc = self.test()
        print(
            "| Epoch {:3d} | test loss {:5.2f}  |  test acc {:5.2f} |                |               |               |".format(
                0,
                test_loss,
                test_acc,
            )
        )
        self.logger.info(
            "| Epoch {:3d} | test loss {:5.2f}  |  test acc {:5.2f} |                |               |               |".format(
                0,
                test_loss,
                test_acc,
            )
        )
        self.finish()

    def finish(self):
        self.logger.info("[*] Training finished!\n\n")
        print("-" * 99)
        print(" [*] Training finished!")
        print(" [*] Please find the training log in results_dir")

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @staticmethod
    def load_encoder(enc_data):
        if enc_data["options"].get("attention_layer_param", 0) == 0:
            enc_data['options']["use_glove"] = False
            model = bilstm_snli(enc_data["options"])
        elif enc_data["options"].get("r", 0) == 0:
            enc_data['options']["use_glove"] = False
            model = attn_bilstm_snli(enc_data["options"])
        else:
            enc_data['options']["use_glove"] = False
            model = struc_attn_snli(enc_data["options"])
        model.load_state_dict(enc_data["model_dict"])
        return model


if __name__ == "__main__":
    args = parse_novelty_conf()
    dataset_conf, optim_conf, model_type, model_conf, sentence_field = get_novelty_conf(
        args
    )
    trainer = Train(
        args, dataset_conf, model_conf, optim_conf, model_type, sentence_field
    )
    trainer.execute()
