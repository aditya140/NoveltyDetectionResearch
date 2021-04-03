import io, os, glob, shutil, re
import six
import requests
import random
import numpy as np

from tqdm import tqdm
import tarfile, zipfile, gzip
from functools import partial
import xml.etree.ElementTree as ET
import json, csv
from collections import defaultdict

import torch
from torchtext import data
from torch.utils.data import Dataset, DataLoader
from torchtext.data import Field, NestedField, LabelField, BucketIterator
from transformers import BertTokenizer, DistilBertTokenizer
from sklearn.model_selection import KFold, StratifiedKFold
import nltk, spacy

from ..utils.download_utils import download_from_url

with open("/root/keys.json",'r') as f:
    apikeys = json.load(f)

"""
Novelty Dataset Base class (torchtext TabularDataset)
"""


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
        if not os.path.exists(os.path.join(path, train)):
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


class APWSJ(NoveltyDataset):
    urls = [
        (
            apikeys['APWSJ_URL'],
            "dataset_aw.zip",
        )
    ]
    dirname = "trec"
    name = "apwsj"

    @classmethod
    def process_apwsj(cls, path):
        AP_path = os.path.join(path, "AP")
        AP_files = glob.glob(os.path.join(AP_path, "*.gz"))
        for i in AP_files:
            with gzip.open(i, "r") as f:
                text = f.read()

            with open(i[:-3], "wb") as f_new:
                f_new.write(text)
            os.remove(i)

        wsj = os.path.join(path, "TREC", "wsj")
        ap = os.path.join(path, "TREC", "AP")
        ap_others = os.path.join(path, "AP")
        wsj_others = os.path.join(path, "WSJ", "wsj_split")
        cmunrf = os.path.join(path, "CMUNRF")

        wsj_files = glob.glob(wsj + "/*")
        ap_files = glob.glob(ap + "/*")

        wsj_other_files = []
        ap_other_files = glob.glob(os.path.join(ap_others, "*"))

        wsj_big = glob.glob(os.path.join(wsj_others, "*"))
        for i in wsj_big:
            for file_path in glob.glob(os.path.join(i, "*")):
                wsj_other_files.append(file_path)

        docs_json = {}
        errors = 0
        for wsj_file in wsj_files:
            with open(wsj_file, "r") as f:
                txt = f.read()
            docs = [
                i.split("<DOC>")[1]
                for i in filter(lambda x: len(x) > 10, txt.split("</DOC>"))
            ]

            for doc in docs:
                try:
                    id = doc.split("<DOCNO>")[1].split("</DOCNO>")[0]
                    text = doc.split("<TEXT>")[1].split("</TEXT>")[0]
                    docs_json[id] = text
                except:
                    errors += 1

        for ap_file in ap_files:
            with open(ap_file, "r", encoding="latin-1") as f:
                txt = f.read()
            docs = [
                i.split("<DOC>")[1]
                for i in filter(lambda x: len(x) > 10, txt.split("</DOC>"))
            ]

            for doc in docs:
                try:
                    id = doc.split("<DOCNO>")[1].split("</DOCNO>")[0]
                    text = doc.split("<TEXT>")[1].split("</TEXT>")[0]
                    docs_json[id] = text
                except:
                    errors += 1

        for wsj_file in wsj_other_files:
            with open(wsj_file, "r") as f:
                txt = f.read()
            docs = [
                i.split("<DOC>")[1]
                for i in filter(lambda x: len(x) > 10, txt.split("</DOC>"))
            ]

            for doc in docs:
                try:
                    id = doc.split("<DOCNO>")[1].split("</DOCNO>")[0]
                    text = doc.split("<TEXT>")[1].split("</TEXT>")[0]
                    docs_json[id] = text
                except:
                    errors += 1

        for ap_file in ap_other_files:
            with open(ap_file, "r", encoding="latin-1") as f:
                txt = f.read()
            docs = [
                i.split("<DOC>")[1]
                for i in filter(lambda x: len(x) > 10, txt.split("</DOC>"))
            ]

            for doc in docs:
                try:
                    id = doc.split("<DOCNO>")[1].split("</DOCNO>")[0]
                    text = doc.split("<TEXT>")[1].split("</TEXT>")[0]
                    docs_json[id] = text
                except:
                    errors += 1

        print("Reading APWSJ dataset, Errors : ", errors)

        docs_json = {k.strip(): v.strip() for k, v in docs_json.items()}

        topic_to_doc_file = os.path.join(cmunrf, "NoveltyData/apwsj.qrels")
        with open(topic_to_doc_file, "r") as f:
            topic_to_doc = f.read()
        topic_doc = [
            (i.split(" 0 ")[1][:-2], i.split(" 0 ")[0])
            for i in topic_to_doc.split("\n")
        ]
        topics = "q101, q102, q103, q104, q105, q106, q107, q108, q109, q111, q112, q113, q114, q115, q116, q117, q118, q119, q120, q121, q123, q124, q125, q127, q128, q129, q132, q135, q136, q137, q138, q139, q141"
        topic_list = topics.split(", ")
        filterd_docid = [(k, v) for k, v in topic_doc if v in topic_list]

        def crawl(red_dict, doc, crawled):
            ans = []
            for cdoc in red_dict[doc]:
                ans.append(cdoc)
                if crawled[cdoc] == 0:
                    try:
                        red_dict[cdoc] = crawl(red_dict, cdoc, crawled)
                        crawled[cdoc] = 1
                        ans += red_dict[cdoc]
                    except:
                        crawled[cdoc] = 1
            return ans

        wf = os.path.join(cmunrf, "redundancy_list_without_partially_redundant.txt")
        redundancy_path = os.path.join(cmunrf, "NoveltyData/redundancy.apwsj.result")
        topics_allowed = "q101, q102, q103, q104, q105, q106, q107, q108, q109, q111, q112, q113, q114, q115, q116, q117, q118, q119, q120, q121, q123, q124, q125, q127, q128, q129, q132, q135, q136, q137, q138, q139, q141"
        topics_allowed = topics_allowed.split(", ")
        red_dict = dict()
        allow_partially_redundant = 1
        for line in open(redundancy_path, "r"):
            tokens = line.split()
            if tokens[2] == "?":
                if allow_partially_redundant == 1:
                    red_dict[tokens[0] + "/" + tokens[1]] = [
                        tokens[0] + "/" + i for i in tokens[3:]
                    ]
            else:
                red_dict[tokens[0] + "/" + tokens[1]] = [
                    tokens[0] + "/" + i for i in tokens[2:]
                ]
        crawled = defaultdict(int)
        for doc in red_dict:
            if crawled[doc] == 0:
                red_dict[doc] = crawl(red_dict, doc, crawled)
                crawled[doc] = 1
        with open(wf, "w") as f:
            for doc in red_dict:
                if doc.split("/")[0] in topics_allowed:
                    f.write(
                        " ".join(
                            doc.split("/") + [i.split("/")[1] for i in red_dict[doc]]
                        )
                        + "\n"
                    )

        write_file = os.path.join(cmunrf, "novel_list_without_partially_redundant.txt")
        topics = topic_list
        doc_topic_dict = defaultdict(list)

        for i in topic_doc:
            doc_topic_dict[i[0]].append(i[1])
        docs_sorted = (
            open(os.path.join(cmunrf, "NoveltyData/apwsj88-90.rel.docno.sorted"), "r")
            .read()
            .splitlines()
        )
        sorted_doc_topic_dict = defaultdict(list)
        for doc in docs_sorted:
            if len(doc_topic_dict[doc]) > 0:
                for t in doc_topic_dict[doc]:
                    sorted_doc_topic_dict[t].append(doc)
        redundant_dict = defaultdict(lambda: defaultdict(int))
        for line in open(
            os.path.join(cmunrf, "redundancy_list_without_partially_redundant.txt"), "r"
        ):
            tokens = line.split()
            redundant_dict[tokens[0]][tokens[1]] = 1
        novel_list = []
        for topic in topics:
            if topic in topics_allowed:
                for i in range(len(sorted_doc_topic_dict[topic])):
                    if redundant_dict[topic][sorted_doc_topic_dict[topic][i]] != 1:
                        if i > 0:
                            # take at most 5 latest docs in case of novel
                            novel_list.append(
                                " ".join(
                                    [topic, sorted_doc_topic_dict[topic][i]]
                                    + sorted_doc_topic_dict[topic][max(0, i - 5) : i]
                                )
                            )
        with open(write_file, "w") as f:
            f.write("\n".join(novel_list))

        # Novel cases
        novel_docs = os.path.join(cmunrf, "novel_list_without_partially_redundant.txt")
        with open(novel_docs, "r") as f:
            novel_doc_list = [i.split() for i in f.read().split("\n")]
        # Redundant cases
        red_docs = os.path.join(
            cmunrf, "redundancy_list_without_partially_redundant.txt"
        )
        with open(red_docs, "r") as f:
            red_doc_list = [i.split() for i in f.read().split("\n")]
        red_doc_list = filter(lambda x: len(x) > 0, red_doc_list)
        novel_doc_list = filter(lambda x: len(x) > 0, novel_doc_list)

        missing_file_log = os.path.join(cmunrf, "missing_log.txt")
        missing_doc_ids = []
        dataset = []
        s_not_found = 0
        t_not_found = 0
        for i in novel_doc_list:
            target_id = i[1]
            source_ids = i[2:]
            if target_id in docs_json.keys():
                data_inst = {}
                data_inst["target_text"] = docs_json[target_id]
                data_inst["source"] = ""
                for source_id in source_ids:
                    if source_id in docs_json.keys():
                        data_inst["source"] += docs_json[source_id] + ". \n"
                    else:
                        missing_doc_ids.append(str(source_id))
                data_inst["DLA"] = "Novel"
            else:
                missing_doc_ids.append(str(target_id))
                # 
            if data_inst["source"] != "":
                dataset.append(data_inst)

        for i in red_doc_list:
            target_id = i[1]
            source_ids = i[2:]
            if target_id in docs_json.keys():
                data_inst = {}
                data_inst["target_text"] = docs_json[target_id]
                data_inst["source"] = ""
                for source_id in source_ids:
                    if source_id in docs_json.keys():
                        data_inst["source"] += docs_json[source_id] + ". \n"
                    else:
                        missing_doc_ids.append(str(source_id))
                data_inst["DLA"] = "Non-Novel"
            else:
                missing_doc_ids.append(str(target_id))
            if data_inst["source"] != "":
                dataset.append(data_inst)
        
        with open(missing_file_log, "w") as f:
            f.write("\n".join(missing_doc_ids))

        dataset_json = {}
        for i in range(len(dataset)):
            dataset_json[i] = dataset[i]
        return dataset_json

    @classmethod
    def process_data(cls, path):
        cmunrf_url = "http://www.cs.cmu.edu/~yiz/research/NoveltyData/CMUNRF1.tar"
        cmunrf_path = os.path.join(path, "CMUNRF1.tar")

        download_from_url(cmunrf_url, cmunrf_path)

        data_zips = [
            (os.path.join(path, "AP.tar"), os.path.join(path, "AP")),
            (os.path.join(path, "trec.zip"), os.path.join(path, "TREC")),
            (os.path.join(path, "wsj_split.zip"), os.path.join(path, "WSJ")),
            (os.path.join(path, "CMUNRF1.tar"), os.path.join(path, "CMUNRF")),
        ]
        for data_zip in data_zips:
            shutil.unpack_archive(data_zip[0], data_zip[1])

        """
        Process APWSJ
        """
        dataset_json = cls.process_apwsj(path)

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, "apwsj.jsonl"), "w") as f:
            f.writelines([json.dumps(i) + "\n" for i in dataset_json.values()])

        # with open(os.path.join(path, "dlnd.jsonl"), "w") as f:
        #     json.dump(list(dataset.values()), f)

    @classmethod
    def splits(
        cls,
        text_field,
        label_field,
        parse_field=None,
        root=".data",
        train="apwsj.jsonl",
        validation=None,
        test=None,
    ):
        return super(APWSJ, cls).splits(
            text_field,
            label_field,
            parse_field=parse_field,
            root=root,
            train=train,
            validation=validation,
            test=test,
        )


class DLND(NoveltyDataset):
    urls = [
        (
            apikeys['DLND_URL'],
            "TAP-DLND-1.0_LREC2018_modified.zip",
        )
    ]
    dirname = "TAP-DLND-1.0_LREC2018_modified"
    name = "dlnd"

    @classmethod
    def process_data(cls, path):
        all_direc = [
            os.path.join(path, direc, direc1)
            for direc in os.listdir(path)
            if os.path.isdir(os.path.join(path, direc))
            for direc1 in os.listdir(os.path.join(path, direc))
        ]

        source_files = [
            [
                os.path.join(direc, "source", file)
                for file in os.listdir(os.path.join(direc, "source"))
                if file.endswith(".txt")
            ]
            for direc in all_direc
        ]
        target_files = [
            [
                os.path.join(direc, "target", file)
                for file in os.listdir(os.path.join(direc, "target"))
                if file.endswith(".txt")
            ]
            for direc in all_direc
        ]
        source_docs = [
            [
                open(file_name, "r", encoding="latin-1")
                .read()
                .encode("ascii", "ignore")
                .decode()
                for file_name in direc
            ]
            for direc in source_files
        ]
        target_docs = [
            [
                open(file_name, "r", encoding="latin-1")
                .read()
                .encode("ascii", "ignore")
                .decode()
                for file_name in direc
            ]
            for direc in target_files
        ]
        data = []
        for i in range(len(target_docs)):
            for j in range(len(target_docs[i])):
                label = [
                    tag.attrib["DLA"]
                    for tag in ET.parse(target_files[i][j][:-4] + ".xml").findall(
                        "feature"
                    )
                    if "DLA" in tag.attrib.keys()
                ][0]
                data.append(
                    [target_docs[i][j]]
                    + [source_docs[i][k] for k in range(len(source_docs[i]))]
                    + ["Novel" if label.lower() == "novel" else "Non-Novel"]
                )
        dataset = []
        for i in data:
            dataset.append(
                {"source":"\n".join(i[1:-1]), "target_text":  i[0], "DLA": i[-1]}
            )

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, "dlnd.jsonl"), "w") as f:
            f.writelines([json.dumps(i) + "\n" for i in dataset])

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


class Webis(NoveltyDataset):
    urls = [
        (
            "https://zenodo.org/record/3251771/files/Webis-CPC-11.zip",
            "Webis-CPC-11.zip",
        )
    ]
    dirname = "Webis-CPC-11"
    name = "webis"

    @classmethod
    def process_data(cls, path):

        original = glob.glob(os.path.join(path, "*original*"))
        metadata = glob.glob(os.path.join(path, "*metadata*"))
        paraphrase = glob.glob(os.path.join(path, "*paraphrase*"))
        assert len(original) == len(metadata) == len(paraphrase)
        ids = [i.split("/")[-1].split("-")[0] for i in original]
        data = {int(i): {} for i in ids}
        to_pop = []
        for id in data.keys():
            org_file = os.path.join(path, f"{id}-original.txt")
            para_file = os.path.join(path, f"{id}-paraphrase.txt")
            meta_file = os.path.join(path, f"{id}-metadata.txt")

            with open(org_file, "r") as f:
                org = f.read()
            with open(para_file, "r") as f:
                par = f.read()
            with open(meta_file, "r") as f:
                text = f.read()
                novel = re.findall("Paraphrase: (.*)", text)[0] == "Yes"
            if len(org) > 10 and len(par) > 10:
                data[id]["source"] = org.replace("\n", "")
                data[id]["target_text"] = par.replace("\n", "")
                data[id]["DLA"] = novel
            else:
                to_pop.append(id)
        for id in to_pop:
            data.pop(id, None)

        dataset = data.values()

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, "webis.jsonl"), "w") as f:
            f.writelines([json.dumps(i) + "\n" for i in dataset])

    @classmethod
    def splits(
        cls,
        text_field,
        label_field,
        parse_field=None,
        root=".data",
        train="webis.jsonl",
        validation=None,
        test=None,
    ):
        return super(Webis, cls).splits(
            text_field,
            label_field,
            parse_field=parse_field,
            root=root,
            train=train,
            validation=validation,
            test=test,
        )


"""
PyTorch Dataset/DataLoader
"""


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


"""
Novelty Dataset Class
"""


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

        if options["doc_field"]:
            self.TEXT_FIELD = self.sentence_field
        else:
            self.TEXT_FIELD = NestedField(
                self.sentence_field,
                tokenize=self.sent_tok,
                fix_length=options["max_num_sent"],
            )
        self.LABEL = LabelField(dtype=torch.long)

        if options["dataset"] == "dlnd":
            dataset = DLND
        if options["dataset"] == "apwsj":
            dataset = APWSJ
        if options["dataset"] == "webis":
            dataset = Webis

        (self.data,) = dataset.splits(self.TEXT_FIELD, self.LABEL)

        if self.options.get("labeled", -1) != -1:
            num_labeled = self.options.get("labeled", False)
            self.dataset_labeled, self.test = self.data.split(
                split_ratio=0.8, stratified=True, random_state=random.getstate()
            )
            data_size = len(self.dataset_labeled)
            percentage = num_labeled / data_size
            self.train, self.dev = self.dataset_labeled.split(
                split_ratio=percentage, stratified=True, random_state=random.getstate()
            )

        else:
            self.train, self.dev, self.test = self.data.split(
                split_ratio=[0.8, 0.1, 0.1],
                stratified=True,
                random_state=random.getstate(),
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
            DLND_Dataset(self.train),
            batch_size=self.options["batch_size"],
            shuffle=True,
        )
        dev_dl = DataLoader(
            DLND_Dataset(self.dev), batch_size=self.options["batch_size"], shuffle=True
        )
        test_dl = DataLoader(
            DLND_Dataset(self.test), batch_size=self.options["batch_size"], shuffle=True
        )
        return train_dl, dev_dl, test_dl

    def get_numpy_data(self):
        def get_numpy(data_iter):
            np_data = {}
            attr_list = ["source", "target", "label"]
            for attr in attr_list:
                data = np.concatenate(
                    [getattr(i, attr).detach().cpu().numpy() for i in data_iter]
                )
                np_data[attr] = data
            src = np.expand_dims(np_data["source"], 1)
            trg = np.expand_dims(np_data["target"], 1)
            inp = np.concatenate([src, trg], axis=1)
            lab = np_data["label"]
            return [inp,lab]

        return (
            get_numpy(self.train_iter),
            get_numpy(self.val_iter),
            get_numpy(self.test_iter),
        )


def novelty_dataset(options, sentence_field=None):
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
