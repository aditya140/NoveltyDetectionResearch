from torchtext import data
from torchtext.data import Field, NestedField, LabelField
import os
import six
import requests
import csv
from tqdm import tqdm
import io
import zipfile
import tarfile
import gzip
import shutil
from functools import partial
import xml.etree.ElementTree as ET
import glob
import json

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
        return data.interleave_keys(len(ex.premise), len(ex.hypothesis))

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
        print(path)
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
                    "DLA": target_set[target]["DLA"],
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


class Novelty:
    def __init__(self):
        self.field = NestedField(Field(), tokenize=lambda x: nltk.sentence_tokenize(x))
