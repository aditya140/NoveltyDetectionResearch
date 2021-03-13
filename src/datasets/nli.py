import os
import sys

import torch
from transformers import BertTokenizer, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.data import (
    Field,
    Iterator,
    TabularDataset,
    LabelField,
    BucketIterator,
    NestedField,
)
from torchtext import datasets
from torchtext.datasets.nli import NLIDataset
from utils.path_utils import makedirs
import pytorch_lightning as pl
from pdb import set_trace

__all__ = ["snli_module", "mnli_module"]


class MultiNLI(NLIDataset):
    urls = ["https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"]
    dirname = "multinli_1.0"
    name = "multinli"

    @classmethod
    def splits(
        cls,
        text_field,
        label_field,
        parse_field=None,
        genre_field=None,
        root=".data",
        train="multinli_1.0_train.jsonl",
        validation="multinli_1.0_dev_matched.jsonl",
        test="multinli_1.0_dev_mismatched.jsonl",
    ):
        extra_fields = {}
        if genre_field is not None:
            extra_fields["genre"] = ("genre", genre_field)

        return super(MultiNLI, cls).splits(
            text_field,
            label_field,
            parse_field=parse_field,
            extra_fields=extra_fields,
            root=root,
            train=train,
            validation=validation,
            test=test,
        )

class NLI_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.fields = self.data.fields
        self._has_char_emb = "premise_char" in self.fields.keys()

    def __len__(self):
        return len(self.data.examples)

    def __getitem__(self, idx):
        premise = (
            self.fields["premise"].process([self.data.examples[idx].premise]).squeeze()
        )
        hypothesis = (
            self.fields["hypothesis"]
            .process([self.data.examples[idx].hypothesis])
            .squeeze()
        )
        label = self.fields["label"].process([self.data.examples[idx].label]).squeeze()
        if not self._has_char_emb:
            return [premise, hypothesis], label
        premise_char = (
            self.fields["premise_char"]
            .process([self.data.examples[idx].premise_char])
            .squeeze()
        )
        hypothesis_char = (
            self.fields["hypothesis_char"]
            .process([self.data.examples[idx].hypothesis_char])
            .squeeze()
        )
        return [premise, hypothesis, premise_char, hypothesis_char], label




"""
SNLI
"""
class SNLI:
    def __init__(self, options):
        self.options = options
        self.tokenizer = options["tokenizer"]

        self.TEXT = Field(
            batch_first=True,
            use_vocab=options["use_vocab"],
            lower=options["lower"],
            preprocessing=options["preprocessing"],
            tokenize=options["tokenize"],
            fix_length=options["max_len"],
            init_token=options["init_token"],
            eos_token=options["eos_token"],
            pad_token=options["pad_token"],
            unk_token=options["unk_token"],
        )
        self.LABEL = LabelField(dtype=torch.long)

        self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)
        if options["use_char_emb"]:
            self.CHAR_TEXT = NestedField(
                Field(
                    pad_token="<p>",
                    tokenize=list,
                    init_token="<s>",
                    eos_token="<e>",
                    batch_first=True,
                    fix_length=options["max_word_len"],
                ),
                fix_length=options["max_len"],
            )
            self.train_char, self.dev_char, self.test_char = datasets.SNLI.splits(
                self.CHAR_TEXT, self.LABEL
            )
            self.CHAR_TEXT.build_vocab(self.train_char)

        if options["use_vocab"]:
            self.TEXT.build_vocab(self.train, self.dev)

        self.LABEL.build_vocab(self.train)

        if options["use_vocab"]:
            vector_cache_loc = ".vector_cache/snli_vectors.pt"
            if os.path.isfile(vector_cache_loc):
                self.TEXT.vocab.vectors = torch.load(vector_cache_loc)
            else:
                self.TEXT.vocab.load_vectors("glove.840B.300d")
                makedirs(os.path.dirname(vector_cache_loc))
                torch.save(self.TEXT.vocab.vectors, vector_cache_loc)

        if options["use_char_emb"]:
            self.train = self.merge_char_dataset(self.train, self.train_char)
            self.dev = self.merge_char_dataset(self.dev, self.dev_char)
            self.test = self.merge_char_dataset(self.test, self.test_char)

        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            (self.train, self.dev, self.test),
            batch_size=options["batch_size"],
            device=options["device"],
        )

    def merge_char_dataset(self, word_dataset, char_dataset):
        assert len(word_dataset) == len(char_dataset)
        for id in range(len(word_dataset)):
            word_ex = word_dataset.examples[id]
            char_ex = char_dataset.examples[id]
            setattr(word_ex, "premise_char", char_ex.premise)
            setattr(word_ex, "hypothesis_char", char_ex.hypothesis)
            word_dataset.fields["premise_char"] = char_dataset.fields["premise"]
            word_dataset.fields["hypothesis_char"] = char_dataset.fields["hypothesis"]
        return word_dataset

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

    def char_padding_idx(self):
        if self.options["use_char_emb"]:
            return self.CHAR_TEXT.vocab.stoi["<p>"]
        else:
            return None

    def out_dim(self):
        return len(self.LABEL.vocab)

    def char_vocab_size(self):
        if self.options["use_char_emb"]:
            return len(self.CHAR_TEXT.vocab)
        else:
            return None

    def labels(self):
        return self.LABEL.vocab.stoi

    def get_dataloaders(self):
        train_dl = DataLoader(
            NLI_Dataset(self.train), batch_size=self.options["batch_size"]
        )
        dev_dl = DataLoader(
            NLI_Dataset(self.dev), batch_size=self.options["batch_size"]
        )
        test_dl = DataLoader(
            NLI_Dataset(self.test), batch_size=self.options["batch_size"]
        )
        return train_dl, dev_dl, test_dl


def snli(options):
    if options["tokenizer"] == "bert" or options["tokenizer"] == "distil_bert":
        
        
        if options["tokenizer"] == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        if options["tokenizer"] == "distil_bert":
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

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

    return SNLI(options)


class SNLIDataModule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.batch_size = conf["batch_size"]

    def prepare_data(self):
        self.data = snli(self.conf)

    def train_dataloader(self):
        return self.data.train_iter

    def val_dataloader(self):
        return self.data.val_iter

    def test_dataloader(self):
        return self.data.test_iter

    def vocab_size(self):
        return self.data.vocab_size()

    def char_vocab_size(self):
        return self.data.char_vocab_size()

    def padding_idx(self):
        return self.data.padding_idx()

    def charpadding_idx(self):
        return self.data.char_padding_idx()

    def out_dim(self):
        return self.data.out_dim()

    def labels(self):
        return self.data.labels()


def snli_module(conf):
    return SNLIDataModule(conf)




"""
MNLI
"""

class MNLI:
    def __init__(self, options):
        self.options = options
        self.tokenizer = options["tokenizer"]

        self.TEXT = Field(
            batch_first=True,
            use_vocab=options["use_vocab"],
            lower=options["lower"],
            preprocessing=options["preprocessing"],
            tokenize=options["tokenize"],
            fix_length=options["max_len"],
            init_token=options["init_token"],
            eos_token=options["eos_token"],
            pad_token=options["pad_token"],
            unk_token=options["unk_token"],
        )
        self.LABEL = LabelField(dtype=torch.float)

        self.train, self.dev, self.test = MultiNLI.splits(self.TEXT, self.LABEL)
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

        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            (self.train, self.dev, self.test),
            batch_size=options["batch_size"],
            device=options["device"],
        )

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

    def get_dataloaders(self):
        train_dl = DataLoader(
            NLI_Dataset(self.train), batch_size=self.options["batch_size"]
        )
        dev_dl = DataLoader(
            NLI_Dataset(self.dev), batch_size=self.options["batch_size"]
        )
        test_dl = DataLoader(
            NLI_Dataset(self.test), batch_size=self.options["batch_size"]
        )
        return train_dl, dev_dl, test_dl


def mnli(options):
    if options["tokenizer"] == "bert" or options["tokenizer"] == "distil_bert":

        if options["tokenizer"] == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        if options["tokenizer"] == "distil_bert":
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

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
        return self.data.train_iter

    def val_dataloader(self):
        return self.data.val_iter

    def test_dataloader(self):
        return self.data.test_iter

    def vocab_size(self):
        return self.data.vocab_size()

    def padding_idx(self):
        return self.data.padding_idx()

    def out_dim(self):
        return self.data.out_dim()

    def labels(self):
        return self.data.labels()


def mnli_module(conf):
    return MNLIDataModule(conf)