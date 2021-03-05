import io, os, glob, shutil, re, bs4, time
import six
import requests
import random
import numpy as np
import itertools

from tqdm import tqdm
import tarfile, zipfile, gzip
from functools import partial
import xml.etree.ElementTree as ET
import json, csv
from collections import defaultdict
import pandas as pd

import torch
from torchtext import data
from torch.utils.data import Dataset, DataLoader
from torchtext.data import Field, NestedField, LabelField, BucketIterator, Example
from transformers import BertTokenizer, DistilBertTokenizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
import nltk, spacy
from nltk.corpus.reader.wordnet import ADJ, ADJ_SAT, ADV, NOUN, VERB
from contextlib import contextmanager

from ..utils.download_utils import download_from_url
from ..utils.topics_reuters import topic_num_map_reuters


from torchtext.datasets import IMDB


import os
import fnmatch
import itertools
from pandas import DataFrame
import re
from html.parser import HTMLParser


class ReutersSGMLParser(HTMLParser):
    """
    Parser of a single SGML file of the reuters-21578 collection.
    """

    def __init__(self, encoding="latin-1"):
        HTMLParser.__init__(self)
        self.encoding = encoding
        self._reset()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""
        self.lewissplit = ""
        self.topics_attribute = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def handle_starttag(self, tag, attributes):
        if tag == "reuters":
            attributes_dict = dict(attributes)
            self.lewissplit = attributes_dict["lewissplit"]
            self.topics_attribute = attributes_dict["topics"]
        elif tag == "title":
            self.in_title = 1
        elif tag == "body":
            self.in_body = 1
        elif tag == "topics":
            self.in_topics = 1
        elif tag == "d":
            self.in_topic_d = 1

    def handle_endtag(self, tag):
        if tag == "reuters":
            self.body = re.sub(r"\s+", r" ", self.body)
            self.docs.append(
                {
                    "title": self.title,
                    "body": self.body,
                    "topics": self.topics,
                    "lewissplit": self.lewissplit,
                    "topics_attribute": self.topics_attribute,
                }
            )
            self._reset()
        elif tag == "title":
            self.in_title = 0
        elif tag == "body":
            self.in_body = 0
        elif tag == "topics":
            self.in_topics = 0
        elif tag == "d":
            self.in_topic_d = 0
            self.topics.append(self.topic_d)
            self.topic_d = ""


class ReutersReader:
    """
    Class used to read the reuters-21578 collection

    :data_path = relative path to the folder containing the source SGML files
    :split = choose between ModApte and ModLewis splits.
    """

    def __init__(self, data_path, split="ModApte"):
        self.data_path = data_path
        self.split = split

    def fetch_documents_generator(self):
        """
        Iterate through all the SGML files and returns a generator cointaining all the documents
        in the router-21578 collection
        """

        for root, _dirnames, filenames in os.walk(self.data_path):
            for filename in fnmatch.filter(filenames, "*.sgm"):
                path = os.path.join(root, filename)
                parser = ReutersSGMLParser()
                for doc in parser.parse(open(path, "rb")):
                    yield doc

    def get_documents(self):
        """
        Returns a dataframe containing one row for each document
        """

        doc_generator = self.fetch_documents_generator()

        if self.split == "ModLewis":
            data = [
                ("{title}\n\n{body}".format(**doc), doc["topics"], doc["lewissplit"])
                for doc in itertools.chain(doc_generator)
                if doc["lewissplit"] != "NOT-USED"
                and doc["topics_attribute"] != "BYPASS"
            ]
        else:
            data = [
                ("{title}\n\n{body}".format(**doc), doc["topics"], doc["lewissplit"])
                for doc in itertools.chain(doc_generator)
                if doc["lewissplit"] != "NOT-USED" and doc["topics_attribute"] == "YES"
            ]

        return DataFrame(data, columns=["text", "topics", "lewissplit"]).to_dict(
            "records"
        )


# https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz


class DocumentDataset(data.Dataset):
    urls = []
    dirname = ""
    name = "novelty"

    @classmethod
    def create_jsonl(cls, path):
        cls.process_data(path)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def download(cls, root, check=None):
        """Download and unzip an online archive (.zip, .gz, or .tgz).

        Arguments:
            root (str): Folder to download data to.
            check (str or None): Folder whose existence indicates
                that the dataset has already been downloaded, or
                None to check the existence of root/{cls.name}.

        Returns:
            str: Path to extracted dataset.
        """
        path = os.path.join(root, cls.name)
        check = path if check is None else check
        if not os.path.isdir(check):
            for url in cls.urls:
                if isinstance(url, tuple):
                    url, filename = url
                else:
                    filename = os.path.basename(url)
                zpath = os.path.join(path, filename)
                if not os.path.isfile(zpath):
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))
                    print("downloading {}".format(filename))
                    download_from_url(url, zpath)
                zroot, ext = os.path.splitext(zpath)
                _, ext_inner = os.path.splitext(zroot)
                if ext == ".zip":
                    with zipfile.ZipFile(zpath, "r") as zfile:
                        print("extracting")
                        zfile.extractall(path)
                # tarfile cannot handle bare .gz files
                elif ext == ".tgz" or ext == ".gz" and ext_inner == ".tar":
                    with tarfile.open(zpath, "r:gz") as tar:
                        dirs = [member for member in tar.getmembers()]
                        tar.extractall(path=path, members=dirs)
                elif ext == ".gz":
                    with gzip.open(zpath, "rb") as gz:
                        with open(zroot, "wb") as uncompressed:
                            shutil.copyfileobj(gz, uncompressed)

        return os.path.join(path, cls.dirname)

    @classmethod
    def splits(
        cls,
        text_field,
        label_field,
        parse_field=None,
        extra_fields={},
        root=".data",
        train="train.jsonl",
        validation="val.jsonl",
        test="test.jsonl",
        top_n=8,
    ):
        """Create dataset objects for splits of the SNLI dataset.

        This is the most flexible way to use the dataset.

        Arguments:
            text_field: The field that will be used for premise and hypothesis
                data.
            label_field: The field that will be used for label data.
            parse_field: The field that will be used for shift-reduce parser
                transitions, or None to not include them.
            extra_fields: A dict[json_key: Tuple(field_name, Field)]
            root: The root directory that the dataset's zip archive will be
                expanded into.
            train: The filename of the train data. Default: 'train.jsonl'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'dev.jsonl'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.jsonl'.
        """

        def binarize_multi_label(label_list):
            mlb = MultiLabelBinarizer()
            mlb.fit(label_list)
            y = mlb.transform(label_list)
            y = np.argmax(y, 1)
            return y, mlb.classes_

        def clean_str(string):
            string = string.lower()
            string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string

        path = cls.download(root)
        if not os.path.exists(os.path.join(path, train)):
            cls.create_jsonl(path)

        if parse_field is None:
            fields = {
                "text": ("text", text_field),
                "label": ("label", label_field),
            }

        jsonl_path = os.path.join(path, train)
        with open(jsonl_path, "r") as f:
            data = [json.loads(i) for i in f.readlines()]
        df = pd.DataFrame.from_records(data)

        df["text"] = df["text"].apply(clean_str)
        df = df[
            df["topics"].apply(
                lambda x: any([i in topic_num_map_reuters.keys() for i in x])
            )
        ]

        # top_n = len(topic_num_map_reuters.keys())
        top_n = 10
        filter_single_class = True

        if filter_single_class:
            df["topics"] = df["topics"].apply(lambda x: [x[0]])

        topic_list = list(itertools.chain(*df["topics"].values))
        unique, counts = np.unique(topic_list, return_counts=True)
        top_n_idx = np.argsort(counts)[-1 * top_n :]
        to_select = unique[top_n_idx]
        df["topic"] = df["topics"].apply(lambda x: [i for i in x if i in to_select])

        y, classes_ = binarize_multi_label(df["topic"])
        label_field.classes_ = classes_
        df["label"] = y.tolist()

        train_data, test_data = (
            df[df["topic"].astype(bool)][df["lewissplit"] == "TRAIN"],
            df[df["topic"].astype(bool)][df["lewissplit"] == "TEST"],
        )

        train_examples = [
            Example.fromlist(i, fields.values())
            for i in train_data[["text", "label"]].values
        ]
        test_examples = [
            Example.fromlist(i, fields.values())
            for i in test_data[["text", "label"]].values
        ]

        return cls(train_examples, fields.values()), cls(test_examples, fields.values())


class Reuters(DocumentDataset):
    urls = [
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz",
            "reuters21578.tar.gz",
        )
    ]
    dirname = "."
    name = "reuters"

    @classmethod
    def process_data(cls, path):

        if not os.path.exists(path):
            os.makedirs(path)

        data = ReutersReader(path)
        dataset_json = data.get_documents()

        with open(os.path.join(path, "reuters.jsonl"), "w") as f:
            f.writelines([json.dumps(i) + "\n" for i in dataset_json])

    @classmethod
    def splits(
        cls,
        text_field,
        label_field,
        parse_field=None,
        root=".data",
        train="reuters.jsonl",
        validation=None,
        test=None,
    ):
        return super(Reuters, cls).splits(
            text_field,
            label_field,
            parse_field=parse_field,
            root=root,
            train=train,
            validation=validation,
            test=test,
        )


class Document:
    def __init__(
        self,
        options,
        sentence_field=None,
    ):
        self.options = options
        if sentence_field == None:
            self.sentence_field = Field(
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
            build_vocab = True
        else:
            self.sentence_field = sentence_field
            build_vocab = False

        if options["sent_tokenizer"] == "spacy":
            import spacy
            from spacy.lang.en import English

            nlp = English()
            nlp.add_pipe(nlp.create_pipe("sentencizer"))

            def sent_tokenize(raw_text):
                doc = nlp(raw_text)
                sentences = [sent.string.strip() for sent in doc.sents]
                return sentences

            self.sent_tok = lambda x: sent_tokenize(x)
        else:
            self.sent_tok = lambda x: nltk.sent_tokenize(x)

        self.TEXT = NestedField(
            self.sentence_field,
            tokenize=self.sent_tok,
            fix_length=options["max_num_sent"],
        )

        if options["dataset"] == "imdb":
            dataset = IMDB
            self.LABEL = LabelField(dtype=torch.long)
            (self.train, self.test) = dataset.splits(self.TEXT, self.LABEL)
            self.LABEL.build_vocab(self.train)

        if options["dataset"] == "reuters":
            dataset = Reuters
            self.LABEL = LABEL = data.Field(
                sequential=False, use_vocab=False, dtype=torch.long
            )
            (self.train, self.test) = dataset.splits(self.TEXT, self.LABEL)

        if build_vocab:
            self.TEXT.build_vocab(self.train, self.test)

        self.train_iter, self.test_iter = BucketIterator.splits(
            (self.train, self.test),
            batch_size=options["batch_size"],
            device=options["device"],
        )

    def vocab_size(self):
        if self.options["use_vocab"]:
            return len(self.TEXT.nesting_field.vocab)
        else:
            return self.tokenizer.vocab_size

    def padding_idx(self):
        if self.options["use_vocab"]:
            return self.TEXT.nesting_field.vocab.stoi[self.options["pad_token"]]
        else:
            return self.options["pad_token"]

    def out_dim(self):
        return len(self.LABEL.vocab)

    def labels(self):
        if hasattr(self.LABEL, "vocab"):
            return self.LABEL.vocab.stoi
        else:
            return {self.LABEL.classes_[i]: i for i in range(len(self.LABEL.classes_))}


def document_dataset(options, sentence_field=None):
    options["use_vocab"] = True
    if sentence_field == None:
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
        options["use_char_emb"] = False

    return Document(options, sentence_field=sentence_field)
