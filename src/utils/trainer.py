import abc, os
import neptune.new as neptune
import datetime
from hyperdash import Experiment
import time
from millify import millify
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_recall_fscore_support

# from torch_lr_finder import LRFinder

from src.defaults import *


### Migrated to new neptune


class Trainer(abc.ABC):
    """[summary]

    Trainer object to train log and save the model file.
    Can be used as a base class to create trainers for specific dataset and saving and loading methods.
    """

    def __init__(
        self,
        args,
        model_conf,
        dataset_conf,
        hparams,
        log_neptune=True,
        log_hyperdash=False,
        **kwargs,
    ):
        """[summary]

        Args:
            args ([type]): args for model
            model_conf (dict): model configuration
            dataset_conf (dict): datset configuration
            hparams (dict): model hyperparamters
            log_neptune (bool, optional): Log to neptune. Defaults to True.
            log_hyperdash (bool, optional): Log to hyperdash. Defaults to True.

        Raises:
            ValueError: [description]

        Trainer object to train log and save the model file.
        Can be used as a base class to create trainers for specific dataset and saving and loading methods.

        Need to implement custom methods load_dataset,load_model,set_optimizers,set_schedulers,save

        """
        print("program execution start: {}".format(datetime.datetime.now()))
        self.scheduler_has_args = False
        self.log_neptune = log_neptune
        self.log_hyperdash = log_hyperdash
        if log_neptune:
            if kwargs.get("neptune_project", None) == None:
                raise ValueError(
                    "Please pass the nepune project name as a keyword argument"
                )
            self.exp = neptune.init(
                project=kwargs["neptune_project"], api_token=NEPTUNE_API, tags=["NEW"]
            )
            self.exp_id = self.exp["sys/id"].fetch()
            self.exp["params/Dataset Conf"].log(dataset_conf)
            self.exp["params/Model Conf"].log(model_conf)
            self.exp["params/Hparams"].log(hparams)

        if log_hyperdash:
            self.hd_exp = Experiment(
                kwargs["neptune_project"], api_key_getter=get_hyperdash_api
            )
            self.hd_exp.param("Dataset Conf", str(dataset_conf))
            self.hd_exp.param("Model Conf", str(model_conf))
            self.hd_exp.param("Hparams", str(hparams))

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
        self.hparams = hparams
        self.load_dataset(dataset_conf, **kwargs)
        self.scheduler_has_args = False
        if not args.folds:
            self.load_model(model_conf, **kwargs)
            model_size = self.count_parameters(self.model)

            print(" [*] Model size : {}".format(model_size))
            print(" [*] Model size : {}".format(millify(model_size, precision=2)))

            self.logger.info(" [*] Model size : {}".format(model_size))
            self.logger.info(
                " [*] Model size (approx) : {}".format(millify(model_size, precision=2))
            )
            if self.log_neptune:
                self.exp["Model size"].log(model_size)
                self.exp["Model size (approx)"].log(millify(model_size, precision=2))

            self.set_optimizers(hparams, **kwargs)
            self.set_schedulers(hparams, **kwargs)
            model_size = self.count_parameters(self.model)
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

    def save(self):
        """
        Implementation on the process of saving model
        """
        pass

    def finish(self):
        """
        Finish Training
        """
        self.logger.info("[*] Training finished!\n\n")
        print("-" * 99)
        print(" [*] Training finished!")
        print(" [*] Please find the training log in results_dir")

        if self.log_hyperdash:
            self.hd_exp.end()

    def train(
        self,
        model,
        optimizer,
        criterion,
        train_iterator,
        log_neptune,
        log_hyperdash,
        **kwargs,
    ):
        """
        Main function to train model.
        Can be called only if load_dataset,load_model,set_optimizers,set_schedulers,save are implemented.

        Args:
            model ([type]): Pytorch model
            optimizer ([type]): Pytorch optimizer
            criterion ([type]): Loss criterion
            train_iterator ([type]): Train Dataset iterator
            log_neptune ([type]): Log to Neptune
            log_hyperdash ([type]): Log to hyperdash

        Raises:
            ValueError: [description]

        Returns:
            train_loss, train_acc: Train Loss, Train accuracy
        """
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
            loss = criterion(answer, label.to(torch.long))
            n_correct += (
                (torch.max(F.softmax(answer, dim=1), 1)[1].view(label.size()) == label)
                .sum()
                .item()
            )
            n_total += batch_size
            n_loss += loss.item()
            if log_neptune:
                self.exp["mertics/Train Loss"].log(loss.item())
            if log_hyperdash:
                self.hd_exp.metric("Train Loss", loss.item(), log=False)
            loss.backward()
            optimizer.step()

        train_loss = n_loss / n_total
        train_acc = 100.0 * n_correct / n_total
        if log_neptune:
            self.exp["mertics/Train Avg Loss"].log(train_loss)
            self.exp["metrics/Train accuracy"].log(train_acc)
        if log_hyperdash:
            self.hd_exp.metric("Train Avg Loss", train_loss, log=False)
            self.hd_exp.metric("Train accuracy", train_acc, log=False)
        return train_loss, train_acc

    def validate(
        self,
        model,
        optimizer,
        criterion,
        val_iterator,
        log_neptune,
        log_hyperdash,
        **kwargs,
    ):
        model.eval()
        if hasattr(val_iterator, "init_epoch") and callable(val_iterator.init_epoch):
            val_iterator.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0
        all_probs, all_preds, all_labels = (
            torch.empty((0, self.label_size)),
            torch.empty((0,)),
            torch.empty((0,)),
        )

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
                loss = criterion(answer, label.to(torch.long))
                prob = F.softmax(answer, dim=1)
                predictions = torch.max(prob, 1)[1].view(label.size())
                n_correct += (predictions == label).sum().item()
                n_total += batch_size
                n_loss += loss.item()

                all_labels = torch.cat([label.cpu(), all_labels], dim=0)
                all_probs = torch.cat([prob.cpu(), all_probs], dim=0)
                if log_neptune:
                    self.exp["Val Loss"].log(loss.item())
                if log_hyperdash:
                    self.hd_exp.metric("Val Loss", loss.item(), log=False)
            val_loss = n_loss / n_total
            val_acc = 100.0 * n_correct / n_total
            if log_neptune:
                self.exp["mertics/Val Avg Loss"].log(val_loss)
                self.exp["mertics/Val Accuracy"].log(val_acc)
            if log_hyperdash:
                self.hd_exp.metric("Val Avg Loss", val_loss, log=False)
                self.hd_exp.metric("Val accuracy", val_acc, log=False)
            return (
                val_loss,
                val_acc,
                (all_probs.tolist(), all_labels.tolist()),
            )

    def test(
        self,
        model,
        criterion,
        test_iterator,
        log_neptune,
        log_hyperdash,
        **kwargs,
    ):
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
        all_probs, all_preds, all_labels = (
            torch.empty((0, self.label_size)),
            torch.empty((0,)),
            torch.empty((0,)),
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_iterator):
                model_inp = [getattr(batch, i) for i in input_attr]
                label = getattr(batch, label_attr)
                batch_size = label.shape[0]

                answer = model(*model_inp)
                loss = criterion(answer, label.to(torch.long))
                prob = F.softmax(answer, dim=1)
                predictions = torch.max(prob, 1)[1].view(label.size())
                n_correct += (predictions == label).sum().item()
                n_total += batch_size
                n_loss += loss.item()
                all_preds = torch.cat([predictions.cpu(), all_preds], dim=0)
                all_labels = torch.cat([label.cpu(), all_labels], dim=0)
                all_probs = torch.cat([prob.cpu(), all_probs], dim=0)

            test_loss = n_loss / n_total
            test_acc = 100.0 * n_correct / n_total
            prec, recall, f1_score, support = precision_recall_fscore_support(
                all_labels, all_preds
            )
            prec = {i: prec[i] for i in range(len(prec))}
            recall = {i: recall[i] for i in range(len(recall))}
            f1_score = {i: f1_score[i] for i in range(len(f1_score))}
            support = {i: support[i] for i in range(len(support))}

            if kwargs.get("secondary_dataset", False):
                if log_neptune:
                    self.exp["mertics/Secondary Test Avg Loss"].log(test_loss)
                    self.exp["mertics/Secondary Test Accuracy"].log(test_acc)
                    self.exp["mertics/Secondary Precision"].log(str(prec))
                    self.exp["mertics/Secondary Recall"].log(str(recall))
                    self.exp["mertics/Secondary F1"].log(str(f1_score))
                return test_loss, test_acc, prec, recall, f1_score

            if log_neptune:
                self.exp["mertics/Test Avg Loss"].log(test_loss)
                self.exp["mertics/Test Accuracy"].log(test_acc)
                self.exp["mertics/Precision"].log(str(prec))
                self.exp["mertics/Recall"].log(str(recall))
                self.exp["mertics/F1"].log(str(f1_score))

            return (
                test_loss,
                test_acc,
                (all_probs.tolist(), all_labels.tolist()),
            )

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def result_checkpoint(self, epoch, train_loss, val_loss, train_acc, val_acc, took):
        if self.best_val_acc is None or val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

            dataset_conf = self.dataset_conf
            if dataset_conf.get("tokenize", False) != False:
                dataset_conf["tokenize"] = None

            torch.save(
                {
                    "accuracy": self.best_val_acc,
                    "options": self.model_conf,
                    "model_dict": self.model.state_dict(),
                    "dataset_conf": dataset_conf,
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
                self.log_neptune,
                self.log_hyperdash,
                **kwargs,
            )
            val_loss, val_acc, (prob, gold) = self.validate(
                self.model,
                self.optimizer,
                self.criterion,
                self.dataset.val_iter,
                self.log_neptune,
                self.log_hyperdash,
                **kwargs,
            )
            if self.scheduler != None:
                if self.scheduler_has_args:
                    self.scheduler.step(val_acc)
                else:
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
            test_loss, test_acc, (prob, gold) = self.test(
                self.model,
                self.criterion,
                self.dataset.test_iter,
                self.log_neptune,
                self.log_hyperdash,
                **kwargs,
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
        if hasattr(self, "secondary_dataset"):
            test_loss, test_acc, prec, recall, f1_score = self.test(
                self.model,
                self.criterion,
                self.secondary_dataset.train_iter,
                self.log_neptune,
                self.log_hyperdash,
                secondary_dataset=True,
                **kwargs,
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
        return test_acc

    def test_folds(self, **kwargs):
        print(" [*] Training starts!")
        print("-" * 99)
        fold_no = 0
        fold_acc = []
        all_probs, all_gold = [], []

        for train_iter, val_iter in self.dataset.iter_folds():

            self.load_model(self.model_conf, **kwargs)
            self.set_optimizers(self.hparams, **kwargs)
            self.set_schedulers(self.hparams, **kwargs)

            fold_no += 1
            start = time.time()
            train_acc_list, test_acc_list, train_loss_list, test_loss_list = (
                [],
                [],
                [],
                [],
            )
            for epoch in range(1, self.args.epochs + 1):
                train_loss, train_acc = self.train(
                    self.model,
                    self.optimizer,
                    self.criterion,
                    self.dataset.train_iter,
                    False,
                    False,
                    **kwargs,
                )
                val_loss, val_acc, (prob, gold) = self.validate(
                    self.model,
                    self.optimizer,
                    self.criterion,
                    self.dataset.val_iter,
                    False,
                    False,
                    **kwargs,
                )
                test_acc_list.append((val_acc, (prob, gold)))
                test_loss_list.append(val_loss)
                train_acc_list.append(train_acc)
                train_loss_list.append(train_loss)

                if self.scheduler != None:
                    self.scheduler.step()

                if self.log_neptune:
                    self.exp["mertics/train acc"].log(train_acc)
                if self.log_hyperdash:
                    self.hd_exp.metric("train acc", train_acc, log=False)

            test_acc, (fold_prob, fold_gold) = max(test_acc_list, key=lambda x: x[0])
            all_probs += fold_prob
            all_gold += fold_gold
            took = time.time() - start
            fold_acc.append(test_acc)

            print(
                "| Fold {:3d}  | train loss {:5.2f} | train acc {:5.2f} | test loss {:5.2f} | test acc {:5.2f} | time: {:5.2f}s |".format(
                    fold_no,
                    min(train_loss_list),
                    max(train_acc_list),
                    min(test_loss_list),
                    test_acc,
                    took,
                )
            )

            if self.log_neptune:
                self.exp["mertics/Fold Accuracy"].log(test_acc)
            if self.log_hyperdash:
                self.hd_exp.metric("Fold Accuracy", test_acc, log=False)

        all_gold = np.array(all_gold, dtype=np.int)
        all_probs = np.array(all_probs)
        all_pred = np.argmax(all_probs, 1)

        with open(
            os.path.join(self.args.results_dir, self.exp_id, "probs.p"), "wb"
        ) as f:
            pickle.dump({"prob": all_probs, "gold": all_gold, "pred": all_pred}, f)

        prec, recall, f1_score, support = precision_recall_fscore_support(
            all_gold, all_pred
        )
        prec = {i: prec[i] for i in range(len(prec))}
        recall = {i: recall[i] for i in range(len(recall))}
        f1_score = {i: f1_score[i] for i in range(len(f1_score))}
        support = {i: support[i] for i in range(len(support))}
        print(
            "| Fold {:3d}  |                | final acc {:5.2f} |                |               |               |".format(
                0,
                sum(fold_acc) / len(fold_acc),
            )
        )
        if self.log_neptune:
            self.exp["mertics/Final Accuracy"].log(sum(fold_acc) / len(fold_acc))
            self.exp["mertics/Final Precision"].log(str(prec))
            self.exp["mertics/Final Recall"].log(str(recall))
            self.exp["mertics/Final F1"].log(str(f1_score))
            self.exp["probs"].upload(
                os.path.join(self.args.results_dir, self.exp_id, "probs.p")
            )

        if self.log_hyperdash:
            self.hd_exp.metric(
                "Final Accuracy", sum(fold_acc) / len(fold_acc), log=False
            )

        self.logger.info(
            "| Fold {:3d}  |                | final acc {:5.2f} |                |               |               |".format(
                0,
                sum(fold_acc) / len(fold_acc),
            )
        )
        fold_acc_avg = sum(fold_acc) / len(fold_acc)
        self.finish()
        return fold_acc_avg
