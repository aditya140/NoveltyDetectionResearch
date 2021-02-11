import sys

sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import datetime
import time

from src.defaults import *
from src.model.nli_models import *
from src.datasets.nli import *
import neptune


class Train:
    def __init__(self, args, dataset_conf, model_conf, hparams, model_type):
        print("program execution start: {}".format(datetime.datetime.now()))

        neptune.init(
            project_qualified_name="aparkhi/NLI",
            api_token=NEPTUNE_API,
        )
        neptune.create_experiment()
        self.args = args
        self.device = get_device(0)
        self.logger = get_logger(self.args, "train")

        self.logger.info("Dataset Conf: {}".format(dataset_conf))
        neptune.log_text("Dataset Conf", str(dataset_conf))

        self.logger.info("Model Conf: {}".format(model_conf))
        neptune.log_text("Model Conf", str(model_conf))
        self.model_conf = model_conf

        self.logger.info("Hparams Conf: {}".format(hparams))
        neptune.log_text("Hparams", str(hparams))

        if dataset_conf["dataset"] == "snli":
            self.dataset = snli_module(dataset_conf)
        elif dataset_conf["dataset"] == "mnli":
            self.dataset = mnli_module(dataset_conf)
        self.dataset.prepare_data()
        self.dataset = self.dataset.data
        neptune.append_tag([dataset_conf["dataset"], model_type])

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

        self.model.to(self.device)
        model_size = self.count_parameters()
        print(" [*] Model size : {}".format(model_size))

        self.logger.info(" [*] Model size : {}".format(model_size))
        neptune.log_text("Model size", str(model_size))

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

        if hparams["optimizer"]["optim"] == "adamw":
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
                },
                "{}/{}/{}/best-{}-{}-params.pt".format(
                    self.args.results_dir,
                    self.args.model_type,
                    self.args.dataset,
                    self.args.model_type,
                    self.args.dataset,
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
            kwargs = {}
            if args.use_char_emb:
                kwargs["char_premise"] = batch.premise_char
                kwargs["char_hypothesis"] = batch.hypothesis_char

            self.opt.zero_grad()
            answer = self.model(batch.premise, batch.hypothesis, **kwargs)
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
                kwargs = {}
                if args.use_char_emb:
                    kwargs["char_premise"] = batch.premise_char
                    kwargs["char_hypothesis"] = batch.hypothesis_char

                answer = self.model(batch.premise, batch.hypothesis, **kwargs)
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
        PATH = "{}/{}/{}/best-{}-{}-params.pt".format(
            self.args.results_dir,
            self.args.model_type,
            self.args.dataset,
            self.args.model_type,
            self.args.dataset,
        )
        model_data = torch.load(PATH)
        self.model.load_state_dict(model_data["model_dict"])
        self.model.eval()
        self.dataset.test_iter.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataset.test_iter):
                kwargs = {}
                if args.use_char_emb:
                    kwargs["char_premise"] = batch.premise_char
                    kwargs["char_hypothesis"] = batch.hypothesis_char

                answer = self.model(batch.premise, batch.hypothesis, **kwargs)
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
        print(" [*] Please find the saved model and training log in results_dir")

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


if __name__ == "__main__":
    args = parse_nli_conf()
    dataset_conf, optim_conf, model_type, model_conf = get_nli_conf(args)
    trainer = Train(args, dataset_conf, model_conf, optim_conf, model_type)
    trainer.execute()
