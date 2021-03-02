import sys

sys.path.append(".")

import joblib
import pickle
import argparse
import os
from joblib import Memory
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.metrics import Accuracy, F1, Recall, Precision
from pytorch_lightning import Callback
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from lang import *
from novelty.cnn.cnn_model import *
from snli.bilstm.bilstm import *
from snli.attn_enc.attn_enc import *
from datamodule import *


location = "./cachedir"
memory = Memory(location, verbose=0)
# memory.clear(warn=False)


@memory.cache
def webis_data_module(Lang, use_nltk):
    data_module = WebisDataModule(batch_size=32)
    data_module.prepare_data(Lang, 100, use_nltk=use_nltk)
    return data_module


@memory.cache
def apwsj_data_module(Lang, use_nltk):
    data_module = APWSJDataModule(batch_size=25)
    print("preparing data")
    data_module.prepare_data(Lang, 100, use_nltk=use_nltk)
    return data_module


@memory.cache
def dlnd_data_module(Lang, use_nltk):
    data_module = DLNDDataModule(batch_size=32)
    data_module.prepare_data(Lang, 100, use_nltk=use_nltk)
    return data_module


@memory.cache
def webis_crossval_datamodule(Lang, use_nltk):
    data_module = WebisDataModule(batch_size=32, cross_val=True)
    print("Started data prep")
    data_module.prepare_data(Lang, 100, use_nltk=use_nltk)
    print("Data Prepared")
    return data_module


@memory.cache
def dlnd_crossval_datamodule(Lang, use_nltk):
    data_module = DLNDDataModule(batch_size=32, cross_val=True)
    print("Started data prep")
    data_module.prepare_data(Lang, 100, use_nltk=use_nltk)
    print("Data Prepared")
    return data_module


@memory.cache
def apwsj_crossval_datamodule(Lang, use_nltk):
    data_module = APWSJDataModule(batch_size=25, cross_val=True)
    print("Started data prep")
    data_module.prepare_data(Lang, 100, use_nltk=use_nltk)
    print("Data Prepared")
    return data_module


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class Novelty_model(pl.LightningModule):
    def __init__(self, model, conf, hparams, trial_set=None):
        super().__init__()
        self.conf = conf
        self.hparams = hparams
        if trial_set != None:
            self.trial_setup(trial_set)
        self.model = model(conf)

        if hasattr(self.conf, "analysis"):
            self.analysis = self.conf.analysis
        else:
            self.analysis = False

        if hasattr(self.conf, "analysisFile"):
            self.analysisFile = self.conf.analysisFile
        else:
            self.analysisFile = "analysisfile.csv"

        self.set_optim("base")

    def forward(self, x0, x1):
        res = self.model.forward(x0, x1)
        return res

    def trial_setup(self, trial_set):
        self.conf, self.hparams = trial_set["trial_func"](
            trial_set["trial"], self.conf, self.hparams
        )

    def set_optim(self, mode):
        if mode == "base":
            self.optimizer_conf = self.hparams.optimizer_base
        else:
            self.optimizer_conf = self.hparams.optimizer_tune

    def configure_optimizers(self):
        # Configure optimizer
        if self.optimizer_conf["optim"].lower() == "AdamW".lower():
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.optimizer_conf["lr"],
            )
        if self.optimizer_conf["optim"].lower() == "Adam".lower():
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.optimizer_conf["lr"],
                weight_decay=self.optimizer_conf["weight_decay"],
            )
        if self.optimizer_conf["optim"].lower() == "sgd".lower():
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.optimizer_conf["lr"],
                momentum=self.optimizer_conf["momentum"],
                weight_decay=self.optimizer_conf["weight_decay"],
            )
        if self.optimizer_conf["optim"].lower() == "adadelta":
            optimizer = optim.Adadelta(
                self.parameters(),
                lr=self.optimizer_conf["lr"],
                rho=self.optimizer_conf["rho"],
                eps=self.optimizer_conf["eps"],
                weight_decay=self.optimizer_conf["weight_decay"],
            )

        # Configure Scheduler
        if self.optimizer_conf["scheduler"] == "plateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            scheduler = {
                "scheduler": lr_scheduler,
                "reduce_on_plateau": True,
                "monitor": "val_checkpoint_on",
            }
            return [optimizer], [scheduler]
        elif self.optimizer_conf["scheduler"] == "lambda":
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda x: 10 ** ((-1) * (x // 6))
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        x0, x1, y, id_ = batch
        opt = self(x0, x1).squeeze(1)
        train_loss = F.cross_entropy(opt, y)
        result = pl.TrainResult(train_loss)
        result.log("training_loss", train_loss)
        return result

    def validation_step(self, batch, batch_idx):
        x0, x1, y, id_ = batch
        opt = self(x0, x1).squeeze(1)
        val_loss = F.cross_entropy(opt, y)
        result = pl.EvalResult(checkpoint_on=val_loss)
        metricF1 = F1(num_classes=2)
        metric = Accuracy(num_classes=2)
        pred = F.softmax(opt)
        test_acc = metric(pred.argmax(1), y)
        test_f1 = metricF1(pred.argmax(1), y)
        result.log("val_acc", test_acc)
        result.log("val_loss", val_loss)
        result.log("val_f1", test_f1)
        return result

    def test_step(self, batch, batch_idx):
        x0, x1, y, id_ = batch
        opt = self(x0, x1).squeeze(1)
        test_loss = F.cross_entropy(opt, y)
        metric = Accuracy(num_classes=2)
        metricF1 = F1(num_classes=2)
        metricRecall = Recall(num_classes=2)
        metricPrecision = Precision(num_classes=2)
        pred = F.softmax(opt)
        test_acc = metric(pred.argmax(1), y)
        test_f1 = metricF1(pred.argmax(1), y)
        test_recall = metricRecall(pred.argmax(1), y)
        test_prec = metricPrecision(pred.argmax(1), y)
        result = pl.EvalResult()
        result.log("test_loss", test_loss)
        result.log("test_acc", test_acc)
        result.log("test_f1", test_f1)
        result.log("test_recall", test_recall)
        result.log("test_prec", test_prec)
        result.log("pred", pred)
        result.log("true", y)
        # Log results if analysis mode is on
        if self.analysis == True:
            result.log("id", id_)

        return result

    def test_end(self, outputs):
        result = pl.EvalResult()
        if self.analysis == True:
            # write results to a file
            ids = outputs["id"].detach().cpu().numpy()
            pred = outputs["pred"].detach().cpu().numpy()[:, 0]
            true = outputs["true"].detach().cpu().numpy()
            df = pd.DataFrame({"id": ids, "pred": pred, "true": true})
            df.to_csv(self.analysisFile)
            df["pred_cls"] = (1 - df["pred"]).apply(lambda x: round(x))
            error_ids = df[df["pred_cls"] != df["true"]].id.values
            with open("./analysis/tempids.txt", "w") as f:
                f.write(",".join([str(i) for i in error_ids]))

        result.log("test_loss", outputs["test_loss"].mean())
        result.log("test_f1", outputs["test_f1"].mean())
        result.log("test_acc", outputs["test_acc"].mean())
        result.log("test_recall", outputs["test_recall"].mean())
        result.log("test_prec", outputs["test_prec"].mean())

        if hasattr(self, "logger"):
            if not isinstance(self.logger, TensorBoardLogger):
                for logger in self.logger:
                    if isinstance(logger, pl.loggers.neptune.NeptuneLogger):

                        y_score = outputs["pred"].detach().cpu().numpy()
                        Y_test = outputs["true"].detach().cpu().numpy()
                        Y_test = np.append(
                            (1 - np.expand_dims(Y_test, axis=0)),
                            np.expand_dims(Y_test, axis=0),
                            0,
                        ).transpose()

                        # Save P-R curve
                        n_classes = 2
                        precision = dict()
                        recall = dict()
                        average_precision = dict()
                        for i in range(n_classes):
                            precision[i], recall[i], _ = precision_recall_curve(
                                Y_test[:, i], y_score[:, i]
                            )
                            average_precision[i] = average_precision_score(
                                Y_test[:, i], y_score[:, i]
                            )
                        precision["micro"], recall["micro"], _ = precision_recall_curve(
                            Y_test.ravel(), y_score.ravel()
                        )
                        average_precision["micro"] = average_precision_score(
                            Y_test, y_score, average="micro"
                        )
                        colors = cycle(
                            [
                                "navy",
                                "turquoise",
                                "darkorange",
                                "cornflowerblue",
                                "teal",
                            ]
                        )
                        plt.figure(figsize=(7, 8))
                        f_scores = np.linspace(0.2, 0.8, num=4)
                        lines = []
                        labels = []
                        for f_score in f_scores:
                            x = np.linspace(0.01, 1)
                            y = f_score * x / (2 * x - f_score)
                            (l,) = plt.plot(
                                x[y >= 0], y[y >= 0], color="gray", alpha=0.2
                            )
                            plt.annotate(
                                "f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02)
                            )
                        lines.append(l)
                        labels.append("iso-f1 curves")
                        (l,) = plt.plot(
                            recall["micro"], precision["micro"], color="gold", lw=2
                        )
                        lines.append(l)
                        labels.append(
                            "micro-average Precision-recall (area = {0:0.2f})"
                            "".format(average_precision["micro"])
                        )
                        for i, color in zip(range(n_classes), colors):
                            (l,) = plt.plot(recall[i], precision[i], color=color, lw=2)
                            lines.append(l)
                            labels.append(
                                "Precision-recall for class {0} (area = {1:0.2f})"
                                "".format(i, average_precision[i])
                            )
                        fig = plt.gcf()
                        fig.subplots_adjust(bottom=0.25)
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel("Recall")
                        plt.ylabel("Precision")
                        plt.title("Extension of Precision-Recall curve to multi-class")
                        plt.legend(lines, labels, loc=(0, -0.38), prop=dict(size=14))
                        logger.experiment.log_image("precision-recall curve", fig)
                        plt.close(fig)
        return result


class Novelty_longformer(pl.LightningModule):
    def __init__(self, model, conf, hparams, trial_set=None):
        super().__init__()
        self.conf = conf
        self.hparams = hparams
        if trial_set != None:
            self.trial_setup(trial_set)
        self.model = model(conf)

        if hasattr(self.conf, "analysis"):
            self.analysis = self.conf.analysis
        else:
            self.analysis = False

        if hasattr(self.conf, "analysisFile"):
            self.analysisFile = self.conf.analysisFile
        else:
            self.analysisFile = "analysisfile.csv"

    def forward(self, x0):
        res = self.model.forward(x0)
        return res

    def trial_setup(self, trial_set):
        self.conf, self.hparams = trial_set["trial_func"](
            trial_set["trial"], self.conf, self.hparams
        )

    def configure_optimizers(self):
        if self.hparams.optim.lower() == "AdamW".lower():
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        if self.hparams.optim.lower() == "Adam".lower():
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        if self.hparams.optim.lower() == "sgd".lower():
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        if self.hparams.scheduler == "plateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=2, factor=0.1
            )
            scheduler = {
                "scheduler": lr_scheduler,
                "reduce_on_plateau": True,
                "monitor": "val_checkpoint_on",
            }
            return [optimizer], [scheduler]
        elif self.hparams.scheduler == "lambda":
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda x: 10 ** ((-1) * (x // 6))
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        x0, y, id_ = batch
        opt = self(x0).squeeze(0)
        train_loss = F.cross_entropy(opt, y)
        result = pl.TrainResult(train_loss)
        result.log("training_loss", train_loss)
        return result

    def validation_step(self, batch, batch_idx):
        x0, y, id_ = batch
        opt = self(x0).squeeze(0)
        val_loss = F.cross_entropy(opt, y)
        result = pl.EvalResult(checkpoint_on=val_loss)
        metricF1 = F1(num_classes=2)
        metric = Accuracy(num_classes=2)
        pred = F.softmax(opt)
        test_acc = metric(pred.argmax(1), y)
        test_f1 = metricF1(pred.argmax(1), y)
        result.log("val_acc", test_acc)
        result.log("val_loss", val_loss)
        result.log("val_f1", test_f1)
        return result

    def test_step(self, batch, batch_idx):
        x0, y, id_ = batch
        opt = self(x0).squeeze(0)
        test_loss = F.cross_entropy(opt, y)
        metric = Accuracy(num_classes=2)
        metricF1 = F1(num_classes=2)
        metricRecall = Recall(num_classes=2)
        metricPrecision = Precision(num_classes=2)
        pred = F.softmax(opt)
        test_acc = metric(pred.argmax(1), y)
        test_f1 = metricF1(pred.argmax(1), y)
        test_recall = metricRecall(pred.argmax(1), y)
        test_prec = metricPrecision(pred.argmax(1), y)
        result = pl.EvalResult()
        result.log("test_loss", test_loss)
        result.log("test_acc", test_acc)
        result.log("test_f1", test_f1)
        result.log("test_recall", test_recall)
        result.log("test_prec", test_prec)
        result.log("pred", pred)
        result.log("true", y)
        # Log results if analysis mode is on
        if self.analysis == True:
            result.log("id", id_)

        return result

    def test_end(self, outputs):
        result = pl.EvalResult()
        if self.analysis == True:
            # write results to a file
            ids = outputs["id"].detach().cpu().numpy()
            pred = outputs["pred"].detach().cpu().numpy()[:, 0]
            true = outputs["true"].detach().cpu().numpy()
            df = pd.DataFrame({"id": ids, "pred": pred, "true": true})
            df.to_csv(self.analysisFile)
            df["pred_cls"] = (1 - df["pred"]).apply(lambda x: round(x))
            error_ids = df[df["pred_cls"] != df["true"]].id.values
            with open("./analysis/tempids.txt", "w") as f:
                f.write(",".join([str(i) for i in error_ids]))

        result.log("test_loss", outputs["test_loss"].mean())
        result.log("test_f1", outputs["test_f1"].mean())
        result.log("test_acc", outputs["test_acc"].mean())
        result.log("test_recall", outputs["test_recall"].mean())
        result.log("test_prec", outputs["test_prec"].mean())

        if hasattr(self, "logger"):
            if not isinstance(self.logger, TensorBoardLogger):
                for logger in self.logger:
                    if isinstance(logger, pl.loggers.neptune.NeptuneLogger):

                        y_score = outputs["pred"].detach().cpu().numpy()
                        Y_test = outputs["true"].detach().cpu().numpy()
                        Y_test = np.append(
                            (1 - np.expand_dims(Y_test, axis=0)),
                            np.expand_dims(Y_test, axis=0),
                            0,
                        ).transpose()

                        # Save P-R curve
                        n_classes = 2
                        precision = dict()
                        recall = dict()
                        average_precision = dict()
                        for i in range(n_classes):
                            precision[i], recall[i], _ = precision_recall_curve(
                                Y_test[:, i], y_score[:, i]
                            )
                            average_precision[i] = average_precision_score(
                                Y_test[:, i], y_score[:, i]
                            )
                        precision["micro"], recall["micro"], _ = precision_recall_curve(
                            Y_test.ravel(), y_score.ravel()
                        )
                        average_precision["micro"] = average_precision_score(
                            Y_test, y_score, average="micro"
                        )
                        colors = cycle(
                            [
                                "navy",
                                "turquoise",
                                "darkorange",
                                "cornflowerblue",
                                "teal",
                            ]
                        )
                        plt.figure(figsize=(7, 8))
                        f_scores = np.linspace(0.2, 0.8, num=4)
                        lines = []
                        labels = []
                        for f_score in f_scores:
                            x = np.linspace(0.01, 1)
                            y = f_score * x / (2 * x - f_score)
                            (l,) = plt.plot(
                                x[y >= 0], y[y >= 0], color="gray", alpha=0.2
                            )
                            plt.annotate(
                                "f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02)
                            )
                        lines.append(l)
                        labels.append("iso-f1 curves")
                        (l,) = plt.plot(
                            recall["micro"], precision["micro"], color="gold", lw=2
                        )
                        lines.append(l)
                        labels.append(
                            "micro-average Precision-recall (area = {0:0.2f})"
                            "".format(average_precision["micro"])
                        )
                        for i, color in zip(range(n_classes), colors):
                            (l,) = plt.plot(recall[i], precision[i], color=color, lw=2)
                            lines.append(l)
                            labels.append(
                                "Precision-recall for class {0} (area = {1:0.2f})"
                                "".format(i, average_precision[i])
                            )
                        fig = plt.gcf()
                        fig.subplots_adjust(bottom=0.25)
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel("Recall")
                        plt.ylabel("Precision")
                        plt.title("Extension of Precision-Recall curve to multi-class")
                        plt.legend(lines, labels, loc=(0, -0.38), prop=dict(size=14))
                        logger.experiment.log_image("precision-recall curve", fig)
                        plt.close(fig)
        return result


class SwitchOptim(Callback):
    def on_train_epoch_start(self, trainer, pl_module):

        if trainer.current_epoch == pl_module.hparams.switch_epoch:
            pl_module.set_optim("tune")
            print("Switching Optimizer at epoch:", trainer.current_epoch)
            optimizers_schedulers = pl_module.configure_optimizers()
            if len(optimizers_schedulers) == 1:
                optimizers = optimizers_schedulers
                trainer.optimizers = optimizers
            else:
                optimizers, schedulers = optimizers_schedulers
                trainer.optimizers = optimizers
                trainer.lr_schedulers = trainer.configure_schedulers(schedulers)
