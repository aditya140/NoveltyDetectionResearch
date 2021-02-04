import os
import sys

import torch
from transformers import BertTokenizer, DistilBertTokenizer

from torchtext.data import Field, Iterator, TabularDataset
from torchtext import datasets
from utils.path_utils import makedirs

from pdb import set_trace

__all__ = ["snli"]


class SNLI:
    def __init__(self, options):
        self.TEXT = Field(
            # lower=options["lower"],
            use_vocab=options["use_vocab"],
            tokenize=options["tokenize"],
            batch_first=True,
            fix_length=options["max_len"],
            init_token=options["init_token"],
            eos_token=options["eos_token"],
            pad_token=options["pad_token"],
            unk_token=options["unk_token"],
            stop_words=options["stopwords"],
        )
        self.LABEL = Field(sequential=False, unk_token=None, is_target=True)

        self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)

        self.TEXT.build_vocab(self.train, self.dev)
        self.LABEL.build_vocab(self.train)

        vector_cache_loc = ".vector_cache/snli_vectors.pt"
        if os.path.isfile(vector_cache_loc):
            self.TEXT.vocab.vectors = torch.load(vector_cache_loc)
        else:
            self.TEXT.vocab.load_vectors("glove.840B.300d")
            makedirs(os.path.dirname(vector_cache_loc))
            torch.save(self.TEXT.vocab.vectors, vector_cache_loc)

        self.train_iter, self.dev_iter, self.test_iter = Iterator.splits(
            (self.train, self.dev, self.test),
            batch_size=options["batch_size"],
            device=options["device"],
        )

    def vocab_size(self):
        return len(self.TEXT.vocab)

    def out_dim(self):
        return len(self.LABEL.vocab)

    def labels(self):
        return self.LABEL.vocab.stoi


def snli(options):
    if options["tokenizer"] == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        sepcial_tokens = tokenizer.special_tokens_map
        options["init_token"] = sepcial_tokens["cls_token"]
        options["pad_token"] = sepcial_tokens["pad_token"]
        options["unk_token"] = sepcial_tokens["unk_token"]
        options["eos_token"] = sepcial_tokens["sep_token"]
        options["use_vocab"] = True
        options["tokenize"] = tokenizer.encode
        options["tokenizer"] = tokenizer

    if options["tokenizer"] == "spacy":
        options["tokenize"] = "spacy"
        options["init_token"] = "<sos>"
        options["unk_token"] = "<unk>"
        options["pad_token"] = "<pad>"
        options["eos_token"] = "<eos>"
        options["use_vocab"] = True

    if options.get("lower", None) == None:
        options["lower"] = True

    if options.get("stopwords", None) == None:
        options["stopwords"] = None

    return SNLI(options)
