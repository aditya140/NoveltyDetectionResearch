import io, os, glob, shutil
import six
import requests
import random
import numpy as np

from tqdm import tqdm
import tarfile, zipfile, gzip
from functools import partial
import xml.etree.ElementTree as ET
import json, csv

import torch
from torchtext import data
from torch.utils.data import Dataset, DataLoader
from torchtext.data import Field, NestedField, LabelField, BucketIterator
from transformers import BertTokenizer, DistilBertTokenizer
from sklearn.model_selection import KFold, StratifiedKFold
import nltk, spacy

from ..utils.download_utils import download_from_url


class NoveltyDataset(data.TabularDataset):
    urls = []
    dirname = ""
    name = "novelty"

    @classmethod
    def create_jsonl(cls, path):
        cls.process_data(path)

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.source), len(ex.target))

    @classmethod
    def download(cls, root, check=None):
        """Download and unzip an online archive (.z ip, .gz, or .tgz).

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
        path = cls.download(root)
        cls.create_jsonl(path)

        if parse_field is None:
            fields = {
                "source": ("source", text_field),
                "target_text": ("target", text_field),
                "DLA": ("label", label_field),
            }

        for key in extra_fields:
            if key not in fields.keys():
                fields[key] = extra_fields[key]

        return super(NoveltyDataset, cls).splits(
            path,
            root,
            train,
            validation,
            test,
            format="json",
            fields=fields,
            filter_pred=lambda ex: ex.label != "-",
        )


class DLND(NoveltyDataset):
    urls = [
        (
            "https://drive.google.com/file/d/1q-P3ReGf-yWnKrhb6XQAuMGo39hXlhYG/view?usp=sharing",
            "TAP-DLND-1.0_LREC2018_modified.zip",
        )
    ]
    dirname = "TAP-DLND-1.0_LREC2018_modified"
    name = "dlnd"

    @classmethod
    def process_data(cls, path):
        def get_sources(source):
            source_meta = [
                "/".join(i.split("/")[:-1])
                + "/"
                + i.split("/")[-1].split(".")[0]
                + ".xml"
                for i in source
            ]
            sources = list(zip(source, source_meta))
            data = []
            for x in sources:
                with open(x[0], mode="r", errors="ignore") as f:
                    source_text = f.read()
                root = ET.parse(x[1]).getroot()
                title = root.findall("feature")[0].get("title")
                eventname = root.findall("feature")[1].get("eventname")
                id = x[0].split("/")[-1].split(".")[0]
                data.append(
                    {
                        "id": id,
                        "eventname": eventname,
                        "title": title,
                        "source_text": source_text,
                    }
                )
            return data

        def filter_labels(x):
            mapping = {
                "Non-Novel": "Non-Novel",
                "Non-Novelvel": "Non-Novel",
                "NovNon-Novelel": "Non-Novel",
                "Novel": "Novel",
                "non-novel": "Non-Novel",
                "novel": "Novel",
            }
            return mapping[x]

        def get_targets(target):
            target_meta = [
                "/".join(i.split("/")[:-1])
                + "/"
                + i.split("/")[-1].split(".")[0]
                + ".xml"
                for i in target
            ]
            targets = list(zip(target, target_meta))
            data = []
            for x in targets:
                with open(x[0], mode="r", errors="ignore") as f:
                    target_text = f.read()
                # with open(x[1],mode='r',errors='ignore') as f:
                #     print(f.read())
                root = ET.parse(x[1]).getroot()
                novel = root.findall("feature")[2].get("DLA")
                src_id = root.findall("feature")[0].get("sourceid").split(",")
                id = x[0].split("/")[-1].split(".")[0]
                eventname = root.findall("feature")[1].get("eventname")
                data.append(
                    {
                        "id": id,
                        "eventname": eventname,
                        "target_text": target_text,
                        "src_id": src_id,
                        "DLA": novel,
                    }
                )
            return data

        categories = glob.glob(os.path.join(path, "*"))
        sources = []
        targets = []
        for cat in categories:
            if os.path.isdir(cat):
                topics = glob.glob(cat + "/*")
                for topic in topics:
                    source = topic + "/source/*.txt"
                    target = topic + "/target/*.txt"
                    event_id = topic + "/EventId.txt"
                    sources += get_sources(glob.glob(source))
                    targets += get_targets(glob.glob(target))

        source_set = {}
        for i in sources:
            source_set[i["id"]] = i
        target_set = {}
        for i in targets:
            target_set[i["id"]] = i

        dataset = {}
        i = 0
        for target in target_set.keys():
            source_text = []
            if len(target_set[target]["src_id"]) > 0 and target_set[target][
                "src_id"
            ] != [""]:
                for src_id in target_set[target]["src_id"]:
                    source_text.append(source_set[src_id]["source_text"])
                dataset[i] = {
                    "target_text": target_set[target]["target_text"],
                    "source": source_text,
                    "DLA": filter_labels(target_set[target]["DLA"]),
                }
                i += 1
        for id_ in dataset.keys():
            dataset[id_]["source"] = "\n".join(dataset[id_]["source"])

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, "dlnd.jsonl"), "w") as f:
            f.writelines([json.dumps(i) + "\n" for i in dataset.values()])

        # with open(os.path.join(path, "dlnd.jsonl"), "w") as f:
        #     json.dump(list(dataset.values()), f)

    @classmethod
    def splits(
        cls,
        text_field,
        label_field,
        parse_field=None,
        root=".data",
        train="dlnd.jsonl",
        validation=None,
        test=None,
    ):
        return super(DLND, cls).splits(
            text_field,
            label_field,
            parse_field=parse_field,
            root=root,
            train=train,
            validation=validation,
            test=test,
        )


class DLND_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.fields = self.data.fields

    def __len__(self):
        return len(self.data.examples)

    def __getitem__(self, idx):
        source = (
            self.fields["source"].process([self.data.examples[idx].source]).squeeze()
        )

        target = (
            self.fields["target"].process([self.data.examples[idx].target]).squeeze()
        )
        label = self.fields["label"].process([self.data.examples[idx].label]).squeeze()

        return [source, target], label


class Novelty:
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

        self.TEXT_FIELD = NestedField(
            self.sentence_field,
            tokenize=self.sent_tok,
            fix_length=options["max_num_sent"],
        )
        self.LABEL = LabelField(dtype=torch.long)

        if options["dataset"] == "dlnd":
            dataset = DLND

        (self.data,) = dataset.splits(self.TEXT_FIELD, self.LABEL)
        self.train, self.dev, self.test = self.data.split(
            split_ratio=[0.8, 0.1, 0.1], stratified=True, random_state=random.getstate()
        )

        self.LABEL.build_vocab(self.train)
        if build_vocab:
            self.TEXT_FIELD.build_vocab(self.train, self.dev)

        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            (self.train, self.dev, self.test),
            batch_size=options["batch_size"],
            device=options["device"],
        )

    def iter_folds(self):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1029)
        train_exs_arr = np.array(self.data.examples)
        labels = np.array([i.label for i in self.data.examples])
        fields = self.data.fields
        for train_idx, test_idx in kf.split(train_exs_arr, y=labels):
            yield (
                BucketIterator(
                    data.Dataset(train_exs_arr[train_idx], fields),
                    batch_size=self.options["batch_size"],
                    device=self.options["device"],
                ),
                BucketIterator(
                    data.Dataset(train_exs_arr[test_idx], fields),
                    batch_size=self.options["batch_size"],
                    device=self.options["device"],
                ),
            )

    def vocab_size(self):
        if self.options["use_vocab"]:
            return len(self.TEXT_FIELD.nesting_field.vocab)
        else:
            return self.tokenizer.vocab_size

    def padding_idx(self):
        if self.options["use_vocab"]:
            return self.TEXT_FIELD.nesting_field.vocab.stoi[self.options["pad_token"]]
        else:
            return self.options["pad_token"]

    def out_dim(self):
        return len(self.LABEL.vocab)

    def labels(self):
        return self.LABEL.vocab.stoi

    def get_dataloaders(self):
        train_dl = DataLoader(
            DLND_Dataset(self.train), batch_size=self.options["batch_size"]
        )
        dev_dl = DataLoader(
            DLND_Dataset(self.dev), batch_size=self.options["batch_size"]
        )
        test_dl = DataLoader(
            DLND_Dataset(self.test), batch_size=self.options["batch_size"]
        )
        return train_dl, dev_dl, test_dl


def dlnd(options, sentence_field=None):
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

    return Novelty(options, sentence_field=sentence_field)
