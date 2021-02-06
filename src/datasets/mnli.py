import os
import sys

import torch
from transformers import BertTokenizer, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.data import Field, Iterator, TabularDataset, LabelField
from torchtext import datasets
from utils.path_utils import makedirs
import pytorch_lightning as pl


from pdb import set_trace

# __all__ = ["mnli", "MNLIDataModule"]


class MNLI:
    def __init__(self, options):
        self.options = options
        self.tokenizer = options["tokenizer"]

        self.TEXT = Field(
            batch_first=True,
            use_vocab=options["use_vocab"],
            preprocessing=options["preprocessing"],
            tokenize=options["tokenize"],
            fix_length=options["max_len"],
            init_token=options["init_token"],
            eos_token=options["eos_token"],
            pad_token=options["pad_token"],
            unk_token=options["unk_token"],
        )
        self.LABEL = LabelField(dtype=torch.float)

        self.train, self.dev, self.test = datasets.MNLI.splits(self.TEXT, self.LABEL)
        if options["use_vocab"]:
            self.TEXT.build_vocab(self.train, self.dev)
        self.LABEL.build_vocab(self.train)

        if options["use_vocab"]:
            vector_cache_loc = ".vector_cache/mnli_vectors.pt"
            if os.path.isfile(vector_cache_loc):
                self.TEXT.vocab.vectors = torch.load(vector_cache_loc)
            else:
                self.TEXT.vocab.load_vectors("glove.840B.300d")
                makedirs(os.path.dirname(vector_cache_loc))
                torch.save(self.TEXT.vocab.vectors, vector_cache_loc)

    def vocab_size(self):
        if self.options["use_vocab"]:
            return len(self.TEXT.vocab)
        else:
            return self.tokenizer.vocab_size

    def padding_idx(self):
        if self.options["use_vocab"]:
            return self.TEXT.vocab.stoi[self.options["pad_token"]]
        else:
            return self.options["pad_token"]

    def out_dim(self):
        return len(self.LABEL.vocab)

    def labels(self):
        return self.LABEL.vocab.stoi


def mnli(options):
    if options["tokenizer"] == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        def tokenize_and_cut(sentence):
            tokens = tokenizer.tokenize(sentence)
            tokens = tokens[: options["max_len"] - 2]
            return tokens

        sepcial_tokens = tokenizer.special_tokens_map
        options["init_token"] = tokenizer.convert_tokens_to_ids(
            sepcial_tokens["cls_token"]
        )
        options["pad_token"] = tokenizer.convert_tokens_to_ids(
            sepcial_tokens["pad_token"]
        )
        options["unk_token"] = tokenizer.convert_tokens_to_ids(
            sepcial_tokens["unk_token"]
        )
        options["eos_token"] = tokenizer.convert_tokens_to_ids(
            sepcial_tokens["sep_token"]
        )
        options["use_vocab"] = False

        options["preprocessing"] = tokenizer.convert_tokens_to_ids

        options["tokenize"] = tokenize_and_cut
        options["tokenizer"] = tokenizer

    if options["tokenizer"] == "spacy":
        options["tokenize"] = "spacy"
        options["init_token"] = "<sos>"
        options["unk_token"] = "<unk>"
        options["pad_token"] = "<pad>"
        options["eos_token"] = "<eos>"
        options["use_vocab"] = True
        options["preprocessing"] = None

    if options.get("lower", None) == None:
        options["lower"] = True

    return MNLI(options)


class MNLIDataModule(pl.LightningDataModule):
    def __init__(self, conf):
        self.conf = conf
        self.batch_size = conf["batch_size"]

    def prepare_data(self):
        self.data = mnli(self.conf)

    def train_dataloader(self):
        return DataLoader(
            self.data.train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
        )

    def vocab_size(self):
        return self.data.vocab_size()

    def padding_idx(self):
        return self.data.padding_idx()

    def out_dim(self):
        return self.data.out_dim()

    def labels(self):
        return self.data.labels()