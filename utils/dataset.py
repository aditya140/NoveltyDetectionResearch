import glob
import re
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import joblib
import random
import torch
import json
from tqdm import tqdm

snli_path = "./dataset/snli/snli_1.0/"

def get_webis_data():
    original = glob.glob("./dataset/novelty/webis/Webis-CPC-11/*original*")
    metadata = glob.glob("./dataset/novelty/webis/Webis-CPC-11/*metadata*")
    paraphrase = glob.glob("./dataset/novelty/webis/Webis-CPC-11/*paraphrase*")
    assert len(original) == len(metadata) == len(paraphrase)
    ids = [i.split("/")[-1].split("-")[0] for i in original]
    data = {int(i): {} for i in ids}
    to_pop = []
    for id in data.keys():
        org_file = f"./dataset/novelty/webis/Webis-CPC-11/{id}-original.txt"
        para_file = f"./dataset/novelty/webis/Webis-CPC-11/{id}-paraphrase.txt"
        meta_file = f"./dataset/novelty/webis/Webis-CPC-11/{id}-metadata.txt"
        with open(org_file, "r") as f:
            org = f.read()
        with open(para_file, "r") as f:
            par = f.read()
        with open(meta_file, "r") as f:
            text = f.read()
            novel = re.findall("Paraphrase: (.*)", text)[0] == "Yes"
        if len(org) > 10 and len(par) > 10:
            data[id]["original"] = org
            data[id]["paraphrase"] = par
            data[id]["isParaphrase"] = novel
        else:
            to_pop.append(id)
    for id in to_pop:
        data.pop(id, None)
    return data

def train_validate_test_split(df, train_percent=0.8, validate_percent=0.1, seed=140):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def get_dlnd_data():
    def get_sources(source):
        source_meta = [
            "/".join(i.split("/")[:-1]) + "/" + i.split("/")[-1].split(".")[0] + ".xml"
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
            "/".join(i.split("/")[:-1]) + "/" + i.split("/")[-1].split(".")[0] + ".xml"
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

    categories = glob.glob("./dataset/novelty/dlnd/TAP-DLND-1.0_LREC2018_modified/*")
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
        if len(target_set[target]["src_id"]) > 0 and target_set[target]["src_id"] != [
            ""
        ]:
            for src_id in target_set[target]["src_id"]:
                source_text.append(source_set[src_id]["source_text"])
            dataset[i] = {
                "target_text": target_set[target]["target_text"],
                "source": source_text,
                "DLA": target_set[target]["DLA"],
            }
            i += 1
    return dataset

def get_apwsj_data():
    apwsj_data_path = "./dataset/apwsj/apwsj_dataset.json"
    with open(apwsj_data_path, "r") as f:
        dataset = json.load(f)
    return dataset

def get_snli_data(data_t):
    snli_file = glob.glob(snli_path + f"*{data_t}*.txt")[0]
    return pd.read_csv(snli_file, sep="\t")



def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)

def get_yelp_data():
    """[summary]
    """
    yelp_path = "./dataset/yelp/yelp_academic_dataset_review.json"
    data = json_reader(yelp_path)

    data_json = {}
    count = 0
    for i in tqdm(data,total=8021122):
        data_json[count] = {"text":i["text"],"label":i["stars"]}
        count+=1
    return data_json



def get_imdb_data():
    """[summary]
    Load IMDB dataset
    Returns:
        [list]: train,test
    """
    __imdb_train_dir__ = "./dataset/imdb/aclImdb/train/"
    __imdb_test_dir__ = "./dataset/imdb/aclImdb/test/"

    neg_train = os.path.join(__imdb_train_dir__, "neg")
    neg_train_files = os.listdir(neg_train)
    pos_train = os.path.join(__imdb_train_dir__, "pos")
    pos_train_files = os.listdir(pos_train)

    neg_test = os.path.join(__imdb_test_dir__, "neg")
    neg_test_files = os.listdir(neg_test)
    pos_test = os.path.join(__imdb_test_dir__, "pos")
    pos_test_files = os.listdir(pos_test)

    pattern = "(\d+)_(\d+)\.txt"

    train = {}
    test = {}
    errors = []
    for i in neg_train_files:
        file_path = os.path.join(neg_train, i)
        try:
            id, rating = tuple(map(int, re.findall(pattern, i)[0]))
            with open(file_path, "r") as f:
                text = f.read()
            train[str(id) + "neg"] = {"text": text, "rating": rating}
        except:
            errors.append(file_path)

    for i in neg_test_files:
        file_path = os.path.join(neg_test, i)
        try:
            id, rating = tuple(map(int, re.findall(pattern, i)[0]))
            with open(file_path, "r") as f:
                text = f.read()
            test[str(id) + "neg"] = {"text": text, "rating": rating}
        except:
            errors.append(file_path)

    for i in pos_train_files:
        file_path = os.path.join(pos_train, i)
        try:
            id, rating = tuple(map(int, re.findall(pattern, i)[0]))
            with open(file_path, "r") as f:
                text = f.read()
            train[str(id) + "pos"] = {"text": text, "rating": rating}
        except:
            errors.append(file_path)

    for i in pos_test_files:
        file_path = os.path.join(pos_test, i)
        try:
            id, rating = tuple(map(int, re.findall(pattern, i)[0]))
            with open(file_path, "r") as f:
                text = f.read()
            test[str(id) + "pos"] = {"text": text, "rating": rating}
        except:
            errors.append(file_path)
    return train, test, errors
