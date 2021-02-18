import abc
import neptune
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.defaults import *


class Trainer(abc.ABC):
    def __init__(
        self, args, model_conf, dataset_conf, hparams, log_neptune=True, **kwargs
    ):
        print("program execution start: {}".format(datetime.datetime.now()))
        self.log_neptune = log_neptune
        if log_neptune:
            if kwargs.get("neptune_project", None) == None:
                raise ValueError(
                    "Please pass the nepune project name as a keyword argument"
                )
            neptune.init(
                project_qualified_name=kwargs["neptune_project"],
                api_token=NEPTUNE_API,
            )
            self.exp = neptune.create_experiment()
            self.exp_id = self.exp.id
            neptune.log_text("Dataset Conf", str(dataset_conf))
            neptune.log_text("Model Conf", str(model_conf))
            neptune.log_text("Hparams", str(hparams))

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = get_logger(self.args, "train", self.exp_id)

        self.logger.info("Dataset Conf: {}".format(dataset_conf))
        self.logger.info("Model Conf: {}".format(model_conf))
        self.logger.info("Hparams Conf: {}".format(hparams))
        self.model_save_path = "{}/{}/model.pt".format(
            self.args.results_dir,
            self.exp_id,
        )

        self.model_conf = model_conf
        self.dataset_conf = dataset_conf
        self.load_dataset(dataset_conf, **kwargs)
        self.load_model(model_conf, **kwargs)
        self.set_optimizers(hparams, **kwargs)
        self.set_schedulers(hparams, **kwargs)
        print("resource preparation done: {}".format(datetime.datetime.now()))

    @abc.abstractmethod
    def load_dataset(self, dataset_conf, **kwargs):
        """
        Implementation on the process of loading dataset.
        """

    @abc.abstractmethod
    def load_model(self, model_conf, **kwargs):
        """
        Implementation on the process of loading dataset.
        """

    @abc.abstractmethod
    def set_optimizers(self, hparams, **kwargs):
        """
        Implementation on the process of loading optimizers.
        """

    @abc.abstractmethod
    def set_schedulers(self, hparams, **kwargs):
        """
        Implementation on the process of loading schedulers.
        """

    @abc.abstractmethod
    def save(self):
        """
        Implementation on the process of saving model
        """

    def finish(self):
        self.logger.info("[*] Training finished!\n\n")
        print("-" * 99)
        print(" [*] Training finished!")
        print(" [*] Please find the training log in results_dir")

    def train(self, model, optimizer, criterion, train_iterator, **kwargs):
        model.train()

        if hasattr(train_iterator, "init_epoch") and callable(
            train_iterator.init_epoch
        ):
            train_iterator.init_epoch()

        n_correct, n_total, n_loss = 0, 0, 0
        if kwargs.get("batch_attr", None) == None:
            raise ValueError(
                """Please provide batch attributes which need to be passed to the model (eg, model inputs, labels)
            batch_attr={model_inp:[source,target],label:label}"""
            )
        else:
            batch_attr = kwargs["batch_attr"]
            input_attr = batch_attr["model_inp"]
            label_attr = batch_attr["label"]

        for batch_idx, batch in enumerate(train_iterator):
            optimizer.zero_grad()
            model_inp = [getattr(batch, i) for i in input_attr]
            label = getattr(batch, label_attr)
            batch_size = label.shape[0]

            answer = model(*model_inp)
            loss = criterion(answer, label)
            n_correct += (
                (torch.max(F.softmax(answer, dim=1), 1)[1].view(label.size()) == label)
                .sum()
                .item()
            )
            n_total += batch_size
            n_loss += loss.item()
            if self.log_neptune:
                neptune.log_metric("Train Loss", loss.item())
            loss.backward()
            optimizer.step()
        train_loss = n_loss / n_total
        train_acc = 100.0 * n_correct / n_total
        if self.log_neptune:
            neptune.log_metric("Train Avg Loss", train_loss)
            neptune.log_metric("Train accuracy", train_acc)
        return train_loss, train_acc

    def validate(self, model, optimizer, criterion, val_iterator, **kwargs):
        model.eval()
        if hasattr(val_iterator, "init_epoch") and callable(val_iterator.init_epoch):
            val_iterator.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0

        if kwargs.get("batch_attr", None) == None:
            raise ValueError(
                """Please provide batch attributes which need to be passed to the model (eg, model inputs, labels)
            batch_attr={model_inp:[source,target],label:label}"""
            )
        else:
            batch_attr = kwargs["batch_attr"]
            input_attr = batch_attr["model_inp"]
            label_attr = batch_attr["label"]

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iterator):
                model_inp = [getattr(batch, i) for i in input_attr]
                label = getattr(batch, label_attr)
                batch_size = label.shape[0]

                answer = model(*model_inp)
                loss = criterion(answer, label)
                n_correct += (
                    (
                        torch.max(F.softmax(answer, dim=1), 1)[1].view(label.size())
                        == label
                    )
                    .sum()
                    .item()
                )
                n_total += batch_size
                n_loss += loss.item()
                if self.log_neptune:
                    neptune.log_metric("Val Loss", loss.item())
            val_loss = n_loss / n_total
            val_acc = 100.0 * n_correct / n_total
            if self.log_neptune:
                neptune.log_metric("Val Avg Loss", val_loss)
                neptune.log_metric("Val Accuracy", val_acc)
            return val_loss, val_acc

    def test(self, model, optimizer, criterion, test_iterator, **kwargs):
        if hasattr(test_iterator, "init_epoch") and callable(test_iterator.init_epoch):
            test_iterator.init_epoch()

        if kwargs.get("batch_attr", None) == None:
            raise ValueError(
                """Please provide batch attributes which need to be passed to the model (eg, model inputs, labels)
            batch_attr={model_inp:[source,target],label:label}"""
            )
        else:
            batch_attr = kwargs["batch_attr"]
            input_attr = batch_attr["model_inp"]
            label_attr = batch_attr["label"]

        model_data = torch.load(self.model_save_path)
        model.load_state_dict(model_data["model_dict"])
        model.eval()

        self.save()

        test_iterator.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_iterator):
                model_inp = [getattr(batch, i) for i in input_attr]
                label = getattr(batch, label_attr)
                batch_size = label.shape[0]

                answer = model(*model_inp)
                loss = criterion(answer, label)
                n_correct += (
                    (
                        torch.max(F.softmax(answer, dim=1), 1)[1].view(label.size())
                        == label
                    )
                    .sum()
                    .item()
                )
                n_total += batch_size
                n_loss += loss.item()
            test_loss = n_loss / n_total
            test_acc = 100.0 * n_correct / n_total
            if self.log_neptune:
                neptune.log_metric("Test Avg Loss", test_loss)
                neptune.log_metric("Test Accuracy", test_acc)
            return test_loss, test_acc

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
                self.model_save_path,
            )

        self.logger.info(
            "| Epoch {:3d} | train loss {:5.2f} | train acc {:5.2f} | val loss {:5.2f} | val acc {:5.2f} | time: {:5.2f}s |".format(
                epoch, train_loss, train_acc, val_loss, val_acc, took
            )
        )

    def fit(self, **kwargs):
        print(" [*] Training starts!")
        print("-" * 99)
        for epoch in range(1, self.args.epochs + 1):
            start = time.time()

            train_loss, train_acc = self.train(
                self.model,
                self.optimizer,
                self.criterion,
                self.dataset.train_iter,
                **kwargs
            )
            val_loss, val_acc = self.validate(
                self.model,
                self.optimizer,
                self.criterion,
                self.dataset.val_iter,
                **kwargs
            )
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

        if hasattr(self.dataset, "test_iter"):
            test_loss, test_acc = self.test(
                self.model,
                self.optimizer,
                self.criterion,
                self.dataset.test_iter,
                **kwargs
            )
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
