import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
import spacy
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from sklearn import preprocessing
from utils.dataset import (
    get_webis_data,
    get_snli_data,
    train_validate_test_split,
    get_dlnd_data,
    get_imdb_data,
    get_apwsj_data,
    get_yelp_data,
)
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import nltk
from tqdm import tqdm


def encode_doc(lang, doc, use_nltk=False):
    if use_nltk:
        return list(
            filter(
                lambda x: x != "" and x != " ",
                [lang.preprocess_sentence(j) for j in nltk.sent_tokenize(doc)],
            )
        )
    else:
        return list(
            filter(
                lambda x: x != "" and x != " ",
                lang.preprocess_sentence(doc).split("."),
            )
        )


class APWSJDataset(Dataset):
    def __init__(self):
        df = pd.DataFrame.from_dict(get_apwsj_data(), orient="index")
        df = df[(df.target != "") & (df.source != "")]
        df.reset_index(drop=True, inplace=True)
        self.data = df.to_dict("index")
        self.data = list(self.data.values())
        self.org = [i["target"] for i in self.data]
        self.par = [i["source"] for i in self.data]
        self.labels = [i["label"] for i in self.data]
        self.labels = torch.tensor(self.labels)

    def encode_lang(self, lang, use_nltk=False):
        self.org = [encode_doc(lang, i, use_nltk=use_nltk) for i in self.org]
        self.par = [encode_doc(lang, i, use_nltk=use_nltk) for i in self.par]
        self.org = [lang.encode_batch(i) for i in self.org]
        self.par = [lang.encode_batch(i) for i in self.par]
        self.max_len = lang.max_len

    def pad_to(self, num_sent):
        pad_arr = [0] * self.max_len
        org_pad_len = [max((num_sent - len(i)), 0) for i in self.org]
        par_pad_len = [max((num_sent - len(i)), 0) for i in self.par]
        self.org = [
            np.append(org, [pad_arr] * org_pad_len[i], axis=0)
            if org_pad_len[i] != 0
            else org[:num_sent]
            for i, org in enumerate(self.org)
        ]
        self.par = [
            np.append(par, [pad_arr] * par_pad_len[i], axis=0)
            if par_pad_len[i] != 0
            else par[:num_sent]
            for i, par in enumerate(self.par)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        org_idx = self.org[idx]
        par_idx = self.par[idx]
        label_idx = self.labels[idx]
        return org_idx, par_idx, label_idx


class WebisDataset(Dataset):
    def __init__(self):
        df = pd.DataFrame.from_dict(get_webis_data(), orient="index")
        df.reset_index(drop=True, inplace=True)
        self.data = df.to_dict("index")
        self.data = list(self.data.values())
        self.org = [i["original"] for i in self.data]
        self.par = [i["paraphrase"] for i in self.data]
        self.label = [i["isParaphrase"] for i in self.data]
        self.le = preprocessing.LabelEncoder()
        self.labels = self.le.fit_transform(self.label)
        self.labels = torch.tensor(self.labels)

    def encode_lang(self, lang, use_nltk=False):
        self.org = [encode_doc(lang, i, use_nltk=use_nltk) for i in self.org]
        self.par = [encode_doc(lang, i, use_nltk=use_nltk) for i in self.par]
        self.org = [lang.encode_batch(i) for i in self.org]
        self.par = [lang.encode_batch(i) for i in self.par]
        self.max_len = lang.max_len

    def pad_to(self, num_sent):
        pad_arr = [0] * self.max_len
        org_pad_len = [max((num_sent - len(i)), 0) for i in self.org]
        par_pad_len = [max((num_sent - len(i)), 0) for i in self.par]
        self.org = [
            np.append(org, [pad_arr] * org_pad_len[i], axis=0)
            if org_pad_len[i] != 0
            else org[:num_sent]
            for i, org in enumerate(self.org)
        ]
        self.par = [
            np.append(par, [pad_arr] * par_pad_len[i], axis=0)
            if par_pad_len[i] != 0
            else par[:num_sent]
            for i, par in enumerate(self.par)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        org_idx = self.org[idx]
        par_idx = self.par[idx]
        label_idx = self.labels[idx]
        return org_idx, par_idx, label_idx


def novelty_collate(batch):
    org = torch.tensor([item[0] for item in batch])
    par = torch.tensor([item[1] for item in batch])
    target = torch.tensor([item[2] for item in batch])
    return [org, par, target]


def checker(txt):
    try:
        float(txt)
        return False
    except:
        return True


class SNLIDataset(Dataset):
    def __init__(self, data_t):
        self.df = get_snli_data(data_t)
        self.df = self.df[
            self.df["gold_label"].isin(["neutral", "entailment", "contradiction"])
        ]
        self.df = self.df[self.df["sentence1"].apply(checker)]
        self.df = self.df[self.df["sentence2"].apply(checker)]

        self.class_labels = self.df["gold_label"].values
        self.hypo = self.df["sentence1"].values
        self.prem = self.df["sentence2"].values
        self.le = preprocessing.LabelEncoder()
        self.labels = self.le.fit_transform(self.class_labels)
        self.labels = torch.tensor(self.labels)
        self.combine = False
        self.char_emb = False

    def encode_lang(self, lang, combine=False):
        self.combine = combine
        if combine:
            self.hypo = lang.encode_batch([self.hypo, self.prem], pair=True)
        else:
            self.hypo = lang.encode_batch(self.hypo)
            self.prem = lang.encode_batch(self.prem)
            if lang.char_emb:
                self.char_emb = True
                self.hypo, self.hypo_char = self.hypo[:, 0], [
                    np.vstack(i) for i in self.hypo[:, 1]
                ]
                self.prem, self.prem_char = self.prem[:, 0], [
                    np.vstack(i) for i in self.prem[:, 1]
                ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.combine:
            return self.hypo[idx], self.labels[idx]
        else:
            if self.char_emb:
                return (
                    self.hypo[idx].astype(int),
                    self.hypo_char[idx].astype(int),
                    self.prem[idx].astype(int),
                    self.prem_char[idx].astype(int),
                    self.labels[idx],
                )
            else:
                return self.hypo[idx], self.prem[idx], self.labels[idx]

    def get_text(
        self,
    ):
        return np.append(self.hypo, self.prem)


def snli_collate(batch):
    org = [item[0] for item in batch]
    par = [item[1] for item in batch]
    target = [item[2] for item in batch]
    return [org, par, target]


class DLNDDataset(Dataset):
    def __init__(self):
        df = pd.DataFrame.from_dict(get_dlnd_data(), orient="index")
        df["source"] = df["source"].apply(lambda x: ". ".join(x))
        df["DLA"] = df["DLA"].apply(
            lambda x: "Non-Novel" if ("non" in x.lower()) else "Novel"
        )
        df["id"] = df.index
        df.reset_index(drop=True, inplace=True)
        df.to_csv("dlnd_data.csv")

        self.data = df.to_dict("index")
        self.data = list(self.data.values())
        self.org = [i["target_text"] for i in self.data]
        self.par = [i["source"] for i in self.data]
        self.label = [i["DLA"] for i in self.data]
        self.id = [int(i["id"]) for i in self.data]
        self.le = preprocessing.LabelEncoder()
        self.labels = self.le.fit_transform(self.label)
        self.labels = torch.tensor(self.labels)
        self.id = torch.tensor(self.id)

    def encode_lang(self, lang, use_nltk=False, combine=False):
        self.combine = combine

        if not self.combine:
            self.org = [encode_doc(lang, i, use_nltk=use_nltk) for i in self.org]
            self.par = [encode_doc(lang, i, use_nltk=use_nltk) for i in self.par]
            self.org = [lang.encode_batch(i) for i in self.org]
            self.par = [lang.encode_batch(i) for i in self.par]
            self.max_len = lang.max_len

        else:
            self.inp = lang.encode_batch([self.org, self.par], pair=True)

    def pad_to(self, num_sent):
        pad_arr = [0] * self.max_len
        org_pad_len = [max((num_sent - len(i)), 0) for i in self.org]
        par_pad_len = [max((num_sent - len(i)), 0) for i in self.par]
        self.org = [
            np.append(org, [pad_arr] * org_pad_len[i], axis=0)
            if org_pad_len[i] != 0
            else org[:num_sent]
            for i, org in enumerate(self.org)
        ]
        self.par = [
            np.append(par, [pad_arr] * par_pad_len[i], axis=0)
            if par_pad_len[i] != 0
            else par[:num_sent]
            for i, par in enumerate(self.par)
        ]

    def getitem__combine(self, idx):
        inp_idx = self.inp[idx]
        label_idx = self.labels[idx]
        id_idx = self.id[idx]
        return inp_idx, label_idx, id_idx

    def getitem__separate(self, idx):
        org_idx = self.org[idx]
        par_idx = self.par[idx]
        label_idx = self.labels[idx]
        id_idx = self.id[idx]
        return org_idx, par_idx, label_idx, id_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.combine:
            return self.getitem__combine(idx)
        else:
            return self.getitem__separate(idx)


# class IMDBDataset(Dataset):
#     def __init__(self):
#         df = pd.DataFrame.from_dict(get_dlnd_data(), orient="index")
#         df["source"] = df["source"].apply(lambda x: ". ".join(x))
#         df["DLA"] = df["DLA"].apply(
#             lambda x: "Non-Novel" if ("non" in x.lower()) else "Novel"
#         )
#         df.reset_index(drop=True, inplace=True)
#         self.data = df.to_dict("index")
#         self.data = list(self.data.values())
#         self.org = [i["target_text"] for i in self.data]
#         self.par = [i["source"] for i in self.data]
#         self.label = [i["DLA"] for i in self.data]
#         self.le = preprocessing.LabelEncoder()
#         self.labels = self.le.fit_transform(self.label)
#         self.labels = torch.tensor(self.labels)

#     def encode_lang(self, lang):
#         self.org = [
#             list(
#                 filter(
#                     lambda x: x != "" and x != " ",
#                     [lang.preprocess_sentence(j) for j in nltk.sent_tokenize(i)],
#                 )
#             )
#             for i in self.org
#         ]
#         self.par = [
#             list(
#                 filter(
#                     lambda x: x != "" and x != " ",
#                     [lang.preprocess_sentence(j) for j in nltk.sent_tokenize(i)],
#                 )
#             )
#             for i in self.par
#         ]
#         self.org = [lang.encode_batch(i) for i in self.org]
#         self.par = [lang.encode_batch(i) for i in self.par]
#         self.max_len = lang.max_len

#     def pad_to(self, num_sent):
#         pad_arr = [0] * self.max_len
#         org_pad_len = [max((num_sent - len(i)), 0) for i in self.org]
#         par_pad_len = [max((num_sent - len(i)), 0) for i in self.par]
#         self.org = [
#             np.append(org, [pad_arr] * org_pad_len[i], axis=0)
#             if org_pad_len[i] != 0
#             else org[:num_sent]
#             for i, org in enumerate(self.org)
#         ]
#         self.par = [
#             np.append(par, [pad_arr] * par_pad_len[i], axis=0)
#             if par_pad_len[i] != 0
#             else par[:num_sent]
#             for i, par in enumerate(self.par)
#         ]

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         org_idx = self.org[idx]
#         par_idx = self.par[idx]
#         label_idx = self.labels[idx]
#         return org_idx, par_idx, label_idx


class IMDBDataset(Dataset):
    def __init__(self):
        train, test, _ = get_imdb_data()
        train_df = pd.DataFrame.from_dict(train, orient="index")
        test_df = pd.DataFrame.from_dict(test, orient="index")
        df = train_df.append(test_df)
        df.reset_index(drop=True, inplace=True)
        self.data = df.to_dict("index")
        self.data = list(self.data.values())
        self.text = [i["text"] for i in self.data]
        self.label = [i["rating"] for i in self.data]
        self.le = preprocessing.LabelEncoder()
        self.label = self.le.fit_transform(self.label)
        self.labels = torch.tensor(self.label)

    def encode_lang(self, lang, use_nltk=False):
        self.text = [encode_doc(lang, i, use_nltk=use_nltk) for i in self.text]
        self.text = [lang.encode_batch(i) for i in self.text]
        self.max_len = lang.max_len

    def pad_to(self, num_sent):
        pad_arr = [0] * self.max_len
        pad_len = [max((num_sent - len(i)), 0) for i in self.text]

        self.text = [
            np.append(par, [pad_arr] * pad_len[i], axis=0)
            if pad_len[i] != 0
            else par[:num_sent]
            for i, par in enumerate(self.text)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_idx = self.text[idx]
        label_idx = self.labels[idx]
        return text_idx, label_idx


class YelpDataset(Dataset):
    def __init__(self):
        data = get_yelp_data()
        data = list(data.values())
        self.text = [i["text"] for i in data]
        self.label = [i["label"] for i in data]
        self.le = preprocessing.LabelEncoder()
        self.label = self.le.fit_transform(self.label)
        self.labels = torch.tensor(self.label)

    def encode_lang(self, lang, use_nltk=False):
        self.text = [encode_doc(lang, i, use_nltk=use_nltk) for i in self.text]
        self.text = [lang.encode_batch(i) for i in tqdm(self.text)]
        self.max_len = lang.max_len

    def pad_to(self, num_sent):
        pad_arr = [0] * self.max_len
        pad_len = [max((num_sent - len(i)), 0) for i in self.text]

        self.text = [
            np.append(par, [pad_arr] * pad_len[i], axis=0)
            if pad_len[i] != 0
            else par[:num_sent]
            for i, par in enumerate(self.text)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_idx = self.text[idx]
        label_idx = self.labels[idx]
        return text_idx, label_idx
