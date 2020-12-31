import pytorch_lightning as pl
from dataloaders import (
    WebisDataset,
    DLNDDataset,
    SNLIDataset,
    IMDBDataset,
    APWSJDataset,
)
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from lang import *
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import torch


class WebisDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, cross_val=False):
        super().__init__()
        self.batch_size = batch_size
        self.cross_val = cross_val

    def set_fold(self, fold_no):
        train_idx, test_idx = self.folds[fold_no]
        self.webis_data_train = Subset(self.webis_data, train_idx)
        self.webis_data_test = Subset(self.webis_data, test_idx)
        self.webis_data_val = Subset(self.webis_data, test_idx)

    def k_fold_split(self):
        data_size = len(self.webis_data)
        kf = KFold(n_splits=10)
        self.folds = []
        for train_index, test_index in kf.split([i for i in range(data_size)]):
            self.folds.append((train_index, test_index))

    def prepare_data(
        self, lang, num_sent, train_size=0.8, test_size=0.1, train_samples=None, seed=42
    ):
        self.webis_data = WebisDataset()
        self.webis_data.encode_lang(lang)
        self.webis_data.pad_to(num_sent)
        if self.cross_val:
            self.k_fold_split()
        else:
            data_size = len(self.webis_data)
            print("Data Size ", data_size)

            if train_samples != None:
                train_size = train_samples
            else:
                train_size = int(train_size * data_size)
            test_size = int(data_size * test_size)
            print("train_samples: ", train_size)
            val_size = data_size - (train_size + test_size)

            (
                self.webis_data_train,
                self.webis_data_val,
                self.webis_data_test,
            ) = random_split(
                self.webis_data,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(seed),
            )

    def train_dataloader(self):
        webis_train = DataLoader(
            self.webis_data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )
        return webis_train

    def val_dataloader(self):
        webis_val = DataLoader(
            self.webis_data_val, batch_size=self.batch_size, shuffle=True, num_workers=6
        )
        return webis_val

    def test_dataloader(self):
        webis_test = DataLoader(
            self.webis_data_test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )
        return webis_test


class DLNDDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, cross_val=False):
        super().__init__()
        self.batch_size = batch_size
        self.cross_val = cross_val

    def set_fold(self, fold_no):
        train_idx, test_idx = self.folds[fold_no]
        self.DLND_data_train = Subset(self.DLND_data, train_idx)
        self.DLND_data_test = Subset(self.DLND_data, test_idx)
        self.DLND_data_val = Subset(self.DLND_data, test_idx)

    def k_fold_split(self):
        data_size = len(self.DLND_data)
        kf = KFold(n_splits=10)
        self.folds = []
        for train_index, test_index in kf.split([i for i in range(data_size)]):
            self.folds.append((train_index, test_index))

    def prepare_data(
        self, lang, num_sent, train_size=0.8, test_size=0.1, train_samples=None, seed=42
    ):
        self.DLND_data = DLNDDataset()
        self.DLND_data.encode_lang(lang)
        self.DLND_data.pad_to(num_sent)
        if self.cross_val:
            self.k_fold_split()
        else:
            data_size = len(self.DLND_data)
            print("Data Size ", data_size)

            if train_samples != None:
                train_size = train_samples
            else:
                train_size = int(train_size * data_size)

            test_size = int(data_size * test_size)
            print("train_samples: ", train_size)
            val_size = data_size - (train_size + test_size)

            (
                self.DLND_data_train,
                self.DLND_data_val,
                self.DLND_data_test,
            ) = random_split(
                self.DLND_data,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        DLND_train = DataLoader(
            self.DLND_data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )
        return DLND_train

    def val_dataloader(self):
        DLND_val = DataLoader(
            self.DLND_data_val, batch_size=self.batch_size, shuffle=True, num_workers=6
        )
        return DLND_val

    def test_dataloader(self):
        DLND_test = DataLoader(
            self.DLND_data_test, batch_size=self.batch_size, shuffle=True, num_workers=6
        )
        return DLND_test


class APWSJDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, cross_val=False):
        super().__init__()
        self.batch_size = batch_size
        self.cross_val = cross_val

    def set_fold(self, fold_no):
        train_idx, test_idx = self.folds[fold_no]
        self.APWSJ_data_train = Subset(self.APWSJ_data, train_idx)
        self.APWSJ_data_test = Subset(self.APWSJ_data, test_idx)
        self.APWSJ_data_val = Subset(self.APWSJ_data, test_idx)

    def k_fold_split(self):
        data_size = len(self.APWSJ_data)
        kf = KFold(n_splits=10)
        self.folds = []
        for train_index, test_index in kf.split([i for i in range(data_size)]):
            self.folds.append((train_index, test_index))

    def prepare_data(
        self, lang, num_sent, train_size=0.8, test_size=0.1, train_samples=None, seed=42
    ):
        self.APWSJ_data = APWSJDataset()
        self.APWSJ_data.encode_lang(lang)
        self.APWSJ_data.pad_to(num_sent)
        if self.cross_val:
            self.k_fold_split()
        else:
            data_size = len(self.APWSJ_data)
            print("Data Size ", data_size)

            if train_samples != None:
                train_size = train_samples
            else:
                train_size = int(train_size * data_size)

            test_size = int(data_size * test_size)
            print("train_samples: ", train_size)
            val_size = data_size - (train_size + test_size)

            (
                self.APWSJ_data_train,
                self.APWSJ_data_val,
                self.APWSJ_data_test,
            ) = random_split(
                self.APWSJ_data,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(140),
            )

    def train_dataloader(self):
        APWSJ_train = DataLoader(
            self.APWSJ_data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )
        return APWSJ_train

    def val_dataloader(self):
        APWSJ_val = DataLoader(
            self.APWSJ_data_val, batch_size=self.batch_size, shuffle=True, num_workers=6
        )
        return APWSJ_val

    def test_dataloader(self):
        APWSJ_test = DataLoader(
            self.APWSJ_data_test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )
        return APWSJ_test


class SNLIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self, lang, lang_conf, combine=False):
        self.snli_data_train = SNLIDataset("train")
        self.snli_data_test = SNLIDataset("test")
        self.snli_data_val = SNLIDataset("dev")
        self.lang_text = np.concatenate(
            (
                self.snli_data_train.get_text(),
                self.snli_data_test.get_text(),
                self.snli_data_val.get_text(),
            ),
            axis=0,
        )
        if lang_conf == "glove":
            self.lang_conf = GloveLangConf(vocab_size=200000)
            self.Lang = LanguageIndex(text=self.lang_text, config=self.lang_conf)
        else:
            self.lang_conf = lang_conf
            self.Lang = lang

        self.snli_data_train.encode_lang(self.Lang, combine=combine)
        self.snli_data_test.encode_lang(self.Lang, combine=combine)
        self.snli_data_val.encode_lang(self.Lang, combine=combine)

    def train_dataloader(self):
        snli_train = DataLoader(
            self.snli_data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )
        return snli_train

    def val_dataloader(self):
        snli_val = DataLoader(
            self.snli_data_val, batch_size=self.batch_size, num_workers=6
        )
        return snli_val

    def test_dataloader(self):
        snli_test = DataLoader(
            self.snli_data_test, batch_size=self.batch_size, num_workers=6
        )
        return snli_test


class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self, lang, num_sent):
        self.IMDB_data = IMDBDataset()
        self.IMDB_data.encode_lang(lang)
        self.IMDB_data.pad_to(num_sent)

        data_size = len(self.IMDB_data)
        train_size = 8 * data_size // 10
        test_size = 1 * data_size // 10
        val_size = data_size - (train_size + test_size)

        (self.IMDB_data_train, self.IMDB_data_val, self.IMDB_data_test,) = random_split(
            self.IMDB_data,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        IMDB_train = DataLoader(
            self.IMDB_data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )
        return IMDB_train

    def val_dataloader(self):
        IMDB_val = DataLoader(
            self.IMDB_data_val, batch_size=self.batch_size, shuffle=True, num_workers=6
        )
        return IMDB_val

    def test_dataloader(self):
        IMDB_test = DataLoader(
            self.IMDB_data_test, batch_size=self.batch_size, shuffle=True, num_workers=6
        )
        return IMDB_test
