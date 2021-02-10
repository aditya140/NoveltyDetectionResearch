import os
import sys

import torch
from transformers import BertTokenizer, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.data import Field, Iterator, TabularDataset, LabelField, BucketIterator
from torchtext import datasets
from utils.path_utils import makedirs
import pytorch_lightning as pl
from pdb import set_trace

__all__ = ["snli_module", "mnli_module"]


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

        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            (self.train, self.dev, self.test),
            batch_size=options["batch_size"],
            device=options["device"],
        )

        if options["use_char_emb"]:
            if options["tokenize"] == "spacy":
                self.max_word_len = options["max_word_len"]
                self.char_vocab = {"": 0}
                self.characterized_words = [
                    [0] * self.max_word_len,
                    [0] * self.max_word_len,
                ]
                self.build_char_vocab()
            else:
                raise Exception("Can use char embeddings only with spacy tokenizer")

    def build_char_vocab(self):
        # for normal words
        for word in self.TEXT.vocab.itos[2:]:
            chars = []
            for c in list(word)[:self.max_word_len]:
                if c not in self.char_vocab:
                    self.char_vocab[c] = len(self.char_vocab)

                chars.append(self.char_vocab[c])

            chars.extend([0] * (self.max_word_len - len(word)))
            self.characterized_words.append(chars)

    def characterize(self, batch):
        """
        :param batch: Pytorch Variable with shape (batch, seq_len)
        :return: Pytorch Variable with shape (batch, seq_len, max_word_len)
        """
        batch = batch.data.cpu().numpy().astype(int).tolist()
        return [[self.characterized_words[w] for w in words] for words in batch]

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

    def char_vocab_size(self):
        return len(self.char_vocab)

    def char_word_len(self):
        return self.max_word_len

    def labels(self):
        return self.LABEL.vocab.stoi


def snli(options):
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

    def char_word_len(self):
        return self.data.char_word_len()

    def padding_idx(self):
        return self.data.padding_idx()

    def out_dim(self):
        return self.data.out_dim()

    def labels(self):
        return self.data.labels()


def snli_module(conf):
    return SNLIDataModule(conf)


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