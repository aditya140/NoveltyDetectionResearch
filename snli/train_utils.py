import sys
sys.path.append(".")

import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from datamodule import *
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.metrics import Accuracy

import pickle
import os
from joblib import Memory
import shutil
import argparse
from lang import *
import joblib
from pytorch_lightning import Callback
from transformers import BertModel, BertTokenizer,DistilBertTokenizer

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

location = "./cachedir"
memory = Memory(location, verbose=0)


@memory.cache
def snli_bert_data_module(batch_size=128,combine = False,tokenizer = None):
    if tokenizer == None:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    if isinstance(tokenizer,str) :
        if tokenizer == 'distilbert-base-uncased':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    bert_lang_conf = BertLangConf(100000)
    Lang = LanguageIndex(None, bert_lang_conf)
    Lang.load_tokenizer(tokenizer)
    data_module = SNLIDataModule(128)
    data_module.prepare_data(Lang, bert_lang_conf,combine=combine)
    return data_module


@memory.cache
def snli_glove_data_module(batch_size=128):
    data_module = SNLIDataModule(batch_size)
    data_module.prepare_data(None, "glove")
    return data_module


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)




class SNLI_base(pl.LightningModule):
    def __init__(self, model, conf, hparams, trial_set=None):
        super().__init__()
        self.conf = conf
        self.hparams = hparams
        print(trial_set)
        if trial_set != None:
            self.trial_setup(trial_set)
        self.model = model(self.conf)
        if self.conf.batch_size != None and self.conf.max_len != None:
            self.example_input_array = (
                torch.randint(0, 10000, (self.conf.batch_size, self.conf.max_len)),
                torch.randint(0, 10000, (self.conf.batch_size, self.conf.max_len)),
            )
        self.set_optim('base')
    
    def set_optim(self,mode):
        if mode == 'base':
            self.optimizer_conf = self.hparams.optimizer_base
        else:
            self.optimizer_conf = self.hparams.optimizer_tune

    def forward(self, x0, x1):
        res = self.model.forward(x0, x1)
        return res

    def trial_setup(self, trial_set):
        print("trial_setup")
        self.conf, self.hparams = trial_set["trial_func"](
            trial_set["trial"], self.conf, self.hparams
        )

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
                optimizer, lr_lambda=lambda x: 10 ** ((-1) * (x // 4))
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]


class SNLI_bert(SNLI_base):
    def __init__(self,model, conf, hparams, trial_set=None):
        super().__init__(model, conf, hparams, trial_set=None)

    def forward(self,x0):
        res = self.model.forward(x0)
        return res

    def training_step(self, batch, batch_idx):
        x0, y = batch
        opt = self(x0).squeeze(0)
        train_loss = F.cross_entropy(opt, y)
        result = pl.TrainResult(train_loss)
        result.log("training_loss", train_loss)
        return result

    def validation_step(self, batch, batch_idx):
        x0, y = batch
        opt = self(x0).squeeze(0)
        val_loss = F.cross_entropy(opt, y)
        result = pl.EvalResult(checkpoint_on=val_loss)
        metric = Accuracy(num_classes=3)
        pred = F.softmax(opt)
        test_acc = metric(pred.argmax(1), y)
        result.log("val_acc", test_acc)
        result.log("val_loss", val_loss)
        return result

    def test_step(self, batch, batch_idx):
        x0, y = batch
        opt = self(x0).squeeze(0)
        test_loss = F.cross_entropy(opt, y)
        result = pl.EvalResult()
        metric = Accuracy(num_classes=3)
        pred = F.softmax(opt)
        test_acc = metric(pred.argmax(1), y)
        result = pl.EvalResult()
        result.log("test_acc", test_acc)
        result.log("test_loss", test_loss)
        return result




class SNLI_model(SNLI_base):
    def __init__(self,model, conf, hparams, trial_set=None):
        super().__init__(model, conf, hparams, trial_set=None)

    def forward(self,x0,x1):
        res = self.model.forward(x0,x1)
        return res

    def training_step(self, batch, batch_idx):
        x0, x1, y = batch
        opt = self(x0, x1).squeeze(0)
        train_loss = F.cross_entropy(opt, y)
        result = pl.TrainResult(train_loss)
        result.log("training_loss", train_loss)
        return result

    def validation_step(self, batch, batch_idx):
        x0, x1, y = batch
        opt = self(x0, x1).squeeze(0)
        val_loss = F.cross_entropy(opt, y)
        result = pl.EvalResult(checkpoint_on=val_loss)
        metric = Accuracy(num_classes=3)
        pred = F.softmax(opt)
        test_acc = metric(pred.argmax(1), y)
        result.log("val_acc", test_acc)
        result.log("val_loss", val_loss)
        return result

    def test_step(self, batch, batch_idx):
        x0, x1, y = batch
        opt = self(x0, x1).squeeze(0)
        test_loss = F.cross_entropy(opt, y)
        result = pl.EvalResult()
        metric = Accuracy(num_classes=3)
        pred = F.softmax(opt)
        test_acc = metric(pred.argmax(1), y)
        result = pl.EvalResult()
        result.log("test_acc", test_acc)
        result.log("test_loss", test_loss)
        return result




class SNLI_struc_attn_model(SNLI_base):
    def __init__(self,model, conf, hparams, trial_set=None):
        super().__init__(model, conf, hparams, trial_set=trial_set)

    def forward(self,x0,x1):
        res = self.model.forward(x0,x1)
        return res

    def training_step(self, batch, batch_idx):
        x0, x1, y = batch
        opt,att0,att1 = self(x0, x1)
        opt = opt.squeeze(0)
        train_loss = F.cross_entropy(opt, y)
        reg_x0 = self.model.penalty_l2(att0)
        reg_x1 = self.model.penalty_l2(att1)
        train_loss = train_loss + self.conf.C *(reg_x0 +reg_x1)
        result = pl.TrainResult(train_loss)
        result.log("training_loss", train_loss)
        return result

    def validation_step(self, batch, batch_idx):
        x0, x1, y = batch
        opt,att0,att1 = self(x0, x1)
        opt = opt.squeeze(0)
        val_loss = F.cross_entropy(opt, y)
        result = pl.EvalResult(checkpoint_on=val_loss)
        metric = Accuracy(num_classes=3)
        pred = F.softmax(opt)
        test_acc = metric(pred.argmax(1), y)
        result.log("val_acc", test_acc)
        result.log("val_loss", val_loss)
        return result

    def test_step(self, batch, batch_idx):
        x0, x1, y = batch
        opt,att0,att1 = self(x0, x1)
        opt = opt.squeeze(0)
        test_loss = F.cross_entropy(opt, y)
        result = pl.EvalResult()
        metric = Accuracy(num_classes=3)
        pred = F.softmax(opt)
        test_acc = metric(pred.argmax(1), y)
        result = pl.EvalResult()
        result.log("test_acc", test_acc)
        result.log("test_loss", test_loss)
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
