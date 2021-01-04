import sys

sys.path.append(".")
import joblib
import pickle
import argparse
from lang import *
from snli.bilstm.bilstm import *
from snli.attn_enc.attn_enc import *
from joblib import Memory
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.metrics import Accuracy,F1
from pytorch_lightning.metrics.regression import RMSE
from pytorch_lightning import Callback
from datamodule import *
import os
from itertools import cycle
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,average_precision_score



location = "./cachedir"
memory = Memory(location, verbose=0)
# memory.clear(warn=False)


@memory.cache
def imdb_data_module(Lang, use_nltk):
    print("Preparing Data Module")
    data_module = IMDBDataModule(batch_size=64)
    data_module.prepare_data(Lang,100, use_nltk=use_nltk)
    print("Prepared Data Module")
    return data_module



@memory.cache
def yelp_data_module(Lang,use_nltk):
    print("Preparing Data Module")
    data_module = YelpDataModule(batch_size=64)
    data_module.prepare_data(Lang,100, use_nltk=use_nltk)
    print("Prepared Data Module")
    return data_module

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)




class Document_model(pl.LightningModule):
    def __init__(self, model, conf, hparams, trial_set=None):
        super().__init__()
        self.conf = conf
        self.hparams = hparams
        if trial_set != None:
            self.trial_setup(trial_set)
        self.model = model(conf)
        self.classes = self.conf.opt_labels
        self.set_optim('base')

    def forward(self, inp):
        res = self.model.forward(inp)
        return res

    def set_optim(self,mode):
        if mode == 'base':
            self.optimizer_conf = self.hparams.optimizer_base
        else:
            self.optimizer_conf = self.hparams.optimizer_tune

    def trial_setup(self, trial_set):
        self.conf, self.hparams = trial_set["trial_func"](
            trial_set["trial"], self.conf, self.hparams
        )

    def configure_optimizers(self):
        # Configure optimizer
        if self.optimizer_conf["optim"].lower() == "AdamW".lower():
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.optimizer_conf["lr"],
                amsgrad = self.optimizer_conf["amsgrad"],
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
                eps=1e-8,
                weight_decay=self.optimizer_conf["weight_decay"],
            )

        if self.optimizer_conf["optim"].lower() == "adagrad":
            optimizer = optim.Adagrad(
                self.parameters(),
                lr=self.optimizer_conf["lr"],
                weight_decay=self.optimizer_conf["weight_decay"],
            )

        if self.optimizer_conf["optim"].lower() == 'rmsprop':
            optimizer = optim.RMSprop(
                self.parameters(),
                lr=self.optimizer_conf["lr"],
                alpha=self.optimizer_conf["alpha"],
                eps=1e-08,
                weight_decay=self.optimizer_conf["weight_decay"],
                momentum=self.optimizer_conf["momentum"],
                centered=False,
            )    
        # Configure Scheduler
        if self.optimizer_conf["scheduler"][0] == "plateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            scheduler = {
                "scheduler": lr_scheduler,
                "reduce_on_plateau": True,
                "monitor": "val_checkpoint_on",
            }
            return [optimizer], [scheduler]
        elif self.optimizer_conf["scheduler"][0] == "lambda":
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda x: self.optimizer_conf["scheduler"][2] ** ((-1) * (x // self.optimizer_conf["scheduler"][1]))
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]


    


class Document_model_reg(Document_model):
    def __init__(self, model, conf, hparams, trial_set=None):
        super().__init__(model,conf,hparams,trial_set=None)
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y=y.to(torch.float32)
        opt = self(x).squeeze(1)
        train_loss = self.loss_fn(opt, y)
        result = pl.TrainResult(train_loss)
        result.log("training_loss", train_loss)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y=y.to(torch.float32)
        opt = self(x).squeeze(1)
        val_loss = self.loss_fn(opt, y)
        result = pl.EvalResult(checkpoint_on=val_loss)
        metric = RMSE()
        test_rmse = metric(opt, y)
        result.log("val_rmse", test_rmse)
        result.log("val_loss", val_loss)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y=y.to(torch.float32)
        opt = self(x).squeeze(1)
        opt_cls = torch.round(opt).to(torch.int64)
        val_loss = self.loss_fn(opt, y)
        result = pl.EvalResult(checkpoint_on=val_loss)
        metric = RMSE()
        test_rmse = metric(opt, y)
        y_cls = y.to(torch.int64)
        metricAcc = Accuracy(num_classes=self.classes)
        metricF1 = F1(num_classes=self.classes)
        test_acc = metricAcc(opt_cls, y_cls)
        test_f1 = metricF1(opt_cls, y_cls)
        result.log("test_f1", test_f1)
        result.log("test_acc", test_acc)
        result.log("test_rmse", test_rmse)
        result.log("test_loss", val_loss)
        return result



class Document_model_clf(Document_model):
    def __init__(self, model, conf, hparams, trial_set=None):
        super().__init__(model,conf,hparams,trial_set=None)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        opt = self(x).squeeze(1)
        train_loss = F.cross_entropy(opt, y)
        result = pl.TrainResult(train_loss)
        result.log("training_loss", train_loss)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        opt = self(x).squeeze(1)
        val_loss = F.cross_entropy(opt, y)
        result = pl.EvalResult(checkpoint_on=val_loss)
        metricF1 = F1(num_classes=self.classes)
        metric = Accuracy(num_classes=self.classes)
        pred = F.softmax(opt)
        test_acc = metric(pred.argmax(1), y)
        test_f1 = metricF1(pred.argmax(1), y)
        result.log("val_acc", test_acc)
        result.log("val_loss", val_loss)
        result.log("val_f1",test_f1)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        opt = self(x).squeeze(1)
        test_loss = F.cross_entropy(opt, y)
        metric = Accuracy(num_classes=self.classes)
        metricF1 = F1(num_classes=self.classes)
        pred = F.softmax(opt)
        test_acc = metric(pred.argmax(1), y)
        test_f1 = metricF1(pred.argmax(1), y)
        result = pl.EvalResult()
        result.log("test_loss", test_loss)
        result.log("test_acc", test_acc)
        result.log("test_f1",test_f1)
        result.log('pred',pred)
        result.log('true',y)
        return result

    def test_end(self,outputs):
        result = pl.EvalResult()
        result.log('test_loss',outputs['test_loss'].mean())
        result.log('test_f1',outputs['test_f1'].mean())
        result.log('test_acc',outputs['test_acc'].mean())

        y_score = outputs["pred"].detach().cpu().numpy()
        Y_test = outputs["true"].detach().cpu()
        # Y_test = np.append((1-np.expand_dims(Y_test, axis=0)),np.expand_dims(Y_test, axis=0),0).transpose()
        Y_test = F.one_hot(Y_test,num_classes=10).numpy()
        # Save P-R curve
        n_classes = self.classes
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:,i],y_score[:,i])
            average_precision[i] = average_precision_score(Y_test[:,i], y_score[:,i])
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
            y_score.ravel())
        average_precision["micro"] = average_precision_score(Y_test, y_score,average="micro")
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        plt.figure(figsize=(14, 16))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                    ''.format(average_precision["micro"]))
        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                        ''.format(i, average_precision[i]))
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(0,-.38), prop=dict(size=14))

        for logger in self.logger:
            if isinstance(logger,pl.loggers.neptune.NeptuneLogger):
                logger.experiment.log_image('precision-recall curve', fig)
        plt.close(fig)

        return result






class Document_model_mixed(Document_model):
    def __init__(self, model, conf, hparams, trial_set=None):
        super().__init__(model,conf,hparams,trial_set=None)
        self.loss_fn_mse = nn.MSELoss()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_reg=y.to(torch.float32)

        reg,clf = self(x)
        reg = reg.squeeze(1)
        clf = clf.squeeze(1)

        train_loss_xe = F.cross_entropy(clf, y)
        train_loss_mse = self.loss_fn_mse(reg, y_reg)

        train_loss = train_loss_xe + train_loss_mse
        result = pl.TrainResult(train_loss)
        result.log("training_loss", train_loss)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_reg=y.to(torch.float32)

        reg,clf = self(x)
        reg = reg.squeeze(1)
        clf = clf.squeeze(1)


        val_loss_xe = F.cross_entropy(clf, y)
        val_loss_mse = self.loss_fn_mse(reg, y_reg)
        val_loss = val_loss_mse+val_loss_xe

        result = pl.EvalResult(checkpoint_on=val_loss)
        metricF1 = F1(num_classes=self.classes)
        metric = Accuracy(num_classes=self.classes)
        pred = F.softmax(clf)
        test_acc = metric(pred.argmax(1), y)
        test_f1 = metricF1(pred.argmax(1), y)
        result.log("val_acc", test_acc)
        result.log("val_loss", val_loss)
        result.log("val_f1",test_f1)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_reg=y.to(torch.float32)

        reg,clf = self(x)
        reg = reg.squeeze(1)
        clf = clf.squeeze(1)
        
        
        test_loss_xe = F.cross_entropy(clf, y)
        test_loss_mse = self.loss_fn_mse(reg, y_reg)
        test_loss = test_loss_xe+test_loss_mse

        metric = Accuracy(num_classes=self.classes)
        metricF1 = F1(num_classes=self.classes)
        pred = F.softmax(clf)
        test_acc = metric(pred.argmax(1), y)
        test_f1 = metricF1(pred.argmax(1), y)
        result = pl.EvalResult()
        result.log("test_loss", test_loss)
        result.log("test_acc", test_acc)
        result.log("test_f1",test_f1)
        result.log('pred',pred)
        result.log('true',y)
        return result

    def test_end(self,outputs):
        result = pl.EvalResult()
        result.log('test_loss',outputs['test_loss'].mean())
        result.log('test_f1',outputs['test_f1'].mean())
        result.log('test_acc',outputs['test_acc'].mean())

        y_score = outputs["pred"].detach().cpu().numpy()
        Y_test = outputs["true"].detach().cpu()
        # Y_test = np.append((1-np.expand_dims(Y_test, axis=0)),np.expand_dims(Y_test, axis=0),0).transpose()
        Y_test = F.one_hot(Y_test,num_classes=10).numpy()
        # Save P-R curve
        n_classes = self.classes
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:,i],y_score[:,i])
            average_precision[i] = average_precision_score(Y_test[:,i], y_score[:,i])
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
            y_score.ravel())
        average_precision["micro"] = average_precision_score(Y_test, y_score,average="micro")
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        plt.figure(figsize=(14, 16))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                    ''.format(average_precision["micro"]))
        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                        ''.format(i, average_precision[i]))
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(0,-.38), prop=dict(size=14))

        for logger in self.logger:
            if isinstance(logger,pl.loggers.neptune.NeptuneLogger):
                logger.experiment.log_image('precision-recall curve', fig)
        plt.close(fig)

        return result




        

class SwitchOptim(Callback):
    def on_train_epoch_start(self, trainer,pl_module):
        
        if trainer.current_epoch == pl_module.hparams.switch_epoch:
            pl_module.set_optim('tune')
            print("Switching Optimizer at epoch:",trainer.current_epoch)
            optimizers_schedulers = pl_module.configure_optimizers()
            if len(optimizers_schedulers)==1:
                optimizers = optimizers_schedulers
                trainer.optimizers = optimizers
            else:
                optimizers, schedulers = optimizers_schedulers
                trainer.optimizers = optimizers
                trainer.lr_schedulers = trainer.configure_schedulers(schedulers)
