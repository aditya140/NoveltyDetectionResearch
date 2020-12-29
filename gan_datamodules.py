import pytorch_lightning as pl
from dataloaders import WebisDataset, DLNDDataset, SNLIDataset, IMDBDataset
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from lang import *
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import torch
from copy import deepcopy


class WebisDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, cross_val=False):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(
        self,
        lang,
        num_sent,
        labeled_samples,
        train_size=0.8,
        test_size=0.1,
    ):
        self.webis_data = WebisDataset()
        self.webis_data.encode_lang(lang)
        self.webis_data.pad_to(num_sent)

        data_size = len(self.webis_data)
        print("Data Size ", data_size)
        train_size = int(train_size * data_size)
        test_size = int(data_size * test_size)
        print("train samples: ", train_size)
        print("Labeled samples: ", labeled_samples)
        val_size = data_size - (train_size + test_size)

        (
            self.webis_data_train,
            self.webis_data_val,
            self.webis_data_test,
        ) = random_split(
            self.webis_data,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # copy of train dataset for labeled dataset
        each_class_size = int(labeled_samples / 2)
        self.webis_train_labeled = deepcopy(self.webis_data_train)
        labels = torch.tensor([y for _, _, y in self.webis_train_labeled])
        indices = torch.arange(len(labels))
        indices = torch.cat(
            [indices[labels == x][:each_class_size] for x in torch.unique(labels)]
        )
        self.webis_train_labeled = Subset(self.webis_train_labeled, indices)

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

    def labeled_dataloader(self):
        webis_labeled = DataLoader(
            self.webis_train_labeled,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )
        return webis_labeled

class DLNDDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, cross_val=False):
        super().__init__()
        self.batch_size = batch_size
        self.cross_val = cross_val


    def prepare_data(
        self,
        lang,
        num_sent,
        labeled_samples,
        train_size=0.8,
        test_size=0.1,
    ):
        self.DLND_data = DLNDDataset()
        self.DLND_data.encode_lang(lang)
        self.DLND_data.pad_to(num_sent)
        
        
        data_size = len(self.DLND_data)
        print("Data Size ", data_size)
        train_size = int(train_size * data_size)
        test_size = int(data_size * test_size)
        print("train samples: ", train_size)
        print("Labeled samples: ", labeled_samples)
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

        # copy of train dataset for labeled dataset
        each_class_size = int(labeled_samples / 2)
        self.DLND_train_labeled = deepcopy(self.DLND_data_train)
        labels = torch.tensor([y for _, _, y in self.DLND_train_labeled])
        indices = torch.arange(len(labels))
        indices = torch.cat(
            [indices[labels == x][:each_class_size] for x in torch.unique(labels)]
        )
        self.DLND_train_labeled = Subset(self.DLND_train_labeled, indices)

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

    def labeled_dataloader(self):
        DLND_labeled = DataLoader(
            self.DLND_train_labeled,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )
        return DLND_labeled

