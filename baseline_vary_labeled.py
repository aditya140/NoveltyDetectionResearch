# %%
import os, pickle, logging, shutil, wget, string, json, nltk, time, sys
from random import shuffle
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import xml.etree.ElementTree as ET
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from nltk import word_tokenize

import neptune.new as neptune


if not os.path.exists("./results/novelty_baseline/"):
    os.makedirs("./results/novelty_baseline/")

logging.basicConfig(
    level=logging.INFO,
    filename="results/novelty_baseline/train_{}.log".format(
        time.strftime("%H:%M:%S", time.gmtime(time.time())),
    ),
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("train")


def download_doc2vecmodel():
    # https://ibm.ent.box.com/s/3f160t4xpuya9an935k84ig465gvymm2
    doc2vec_url = "https://ibm.box.com/shared/static/3f160t4xpuya9an935k84ig465gvymm2"
    wget.download(doc2vec_url, "enwiki_dbow.tgz")


def unpack_doc2vecmodel():
    if os.path.exists("enwiki_dbow.tgz"):
        shutil.unpack_archive("./enwiki_dbow.tgz", "enwiki_dbow")


# if not os.path.exists("./enwiki_dbow/enwiki_dbow"):
#     download_doc2vecmodel()
#     unpack_doc2vecmodel()

# model = Doc2Vec.load("./enwiki_dbow/enwiki_dbow/doc2vec.bin")
stopwords = list(string.punctuation) + list(set(stopwords.words("english")))
tfidf_vec = TfidfVectorizer(
    decode_error="ignore",
    lowercase=False,
    stop_words=stopwords,
    sublinear_tf=True,
    smooth_idf=True,
)
tfidf_vec1 = TfidfVectorizer(
    decode_error="ignore", lowercase=False, stop_words=stopwords, smooth_idf=True
)
count_vec = CountVectorizer(
    decode_error="ignore", lowercase=False, stop_words=stopwords
)


def make_cv_10_fold(labels):
    cv = [None] * n_cases
    pos_rows = []
    neg_rows = []
    for n, l in enumerate(labels):
        if l == 1:
            pos_rows.append(n)
        elif l == 0:
            neg_rows.append(n)
    shuffle(pos_rows)
    shuffle(neg_rows)
    for i in range(10):
        for n in pos_rows[
            int((len(pos_rows) * i) / float(10)) : int(
                (len(pos_rows) * (i + 1)) / float(10)
            )
        ]:
            cv[n] = i
    for i in range(10):
        for n in neg_rows[
            int((len(neg_rows) * i) / float(10)) : int(
                (len(neg_rows) * (i + 1)) / float(10)
            )
        ]:
            cv[n] = i
    ##
    cv_dict = dict()
    for i in range(len(cv)):
        try:
            cv_dict[cv[i]].append(labels[i])
        except:
            cv_dict[cv[i]] = [labels[i]]
    print([len(cv_dict[key]) for key in cv_dict.keys()])
    print([sum(cv_dict[key]) for key in cv_dict.keys()])
    ##
    return cv


def vary_labeled(features):
    global labels, labeled_sizes
    class_order = np.unique(labels)
    index_per_class = []
    for i in class_order:
        index_per_class.append(np.argwhere(np.array(labels) == i).reshape(1, -1)[0])

    vary_acc = []
    train_ids = []
    for labeled_size in labeled_sizes:
        classes_ = len(class_order)
        per_class = labeled_size // classes_
        for class_no in index_per_class:
            train_ids += random.choices(class_no, k=per_class)

        lr = LogisticRegression(verbose=0, n_jobs=-1)
        predictions = [None] * len(labels)
        probs = [None] * len(labels)
        trainX = []
        trainY = []
        testX = []
        testY = []

        for i, e in enumerate(labels):
            if i in train_ids:
                trainX.append(features[i])
                trainY.append(labels[i])
            else:
                testX.append(features[i])
                testY.append(labels[i])

        trainX = np.array(trainX)
        if len(trainX.shape) == 1:
            trainX = trainX.reshape(-1, 1)
        testX = np.array(testX)
        if len(testX.shape) == 1:
            testX = testX.reshape(-1, 1)
        lr.fit(trainX, trainY)
        prob = lr.predict_proba(testX)
        preds = [class_order[i] for i in np.argmax(prob, axis=1)]
        prob = prob.tolist()

        gold = np.array(testY)
        predictions = np.array(preds)
        cf = confusion_matrix(gold, predictions, labels=[0, 1])
        acc = accuracy_score(gold, predictions)
        p, r, f, _ = precision_recall_fscore_support(gold, predictions, labels=[0, 1])
        vary_acc.append(acc)
    return vary_acc


def train_lr(features):
    global labels, cv
    class_order = np.unique(labels)
    lr = LogisticRegression(verbose=0, n_jobs=-1)
    predictions = [None] * len(labels)
    probs = [None] * len(labels)
    for curr_cv in range(10):
        trainX = []
        trainY = []
        testX = []
        testY = []
        for i in range(len(cv)):
            if cv[i] == curr_cv:
                testX.append(features[i])
                testY.append(labels[i])
            else:
                trainX.append(features[i])
                trainY.append(labels[i])
        trainX = np.array(trainX)
        if len(trainX.shape) == 1:
            trainX = trainX.reshape(-1, 1)
        testX = np.array(testX)
        if len(testX.shape) == 1:
            testX = testX.reshape(-1, 1)
        lr.fit(trainX, trainY)
        prob = lr.predict_proba(testX)
        preds = [class_order[i] for i in np.argmax(prob, axis=1)]
        prob = prob.tolist()
        correct_indices = [i for i in range(len(cv)) if cv[i] == curr_cv]
        for i in range(len(correct_indices)):
            predictions[correct_indices[i]] = preds[i]
            probs[correct_indices[i]] = prob[i]
    gold = np.array(labels)
    predictions = np.array(predictions)
    cf = confusion_matrix(gold, predictions, labels=[0, 1])
    acc = accuracy_score(gold, predictions)
    p, r, f, _ = precision_recall_fscore_support(gold, predictions, labels=[0, 1])
    print("\nConfusion matrix:\n" + str(cf))
    print("\nAccuracy: " + str(acc))
    print("\nClass wise precisions: " + str(p))
    print("Class wise recalls: " + str(r))
    print("Class wise fscores: " + str(f))

    logger.info("\nConfusion matrix:\n" + str(cf))
    logger.info("\nAccuracy: " + str(acc))
    logger.info("\nClass wise precisions: " + str(p))
    logger.info("Class wise recalls: " + str(r))
    logger.info("Class wise fscores: " + str(f))
    return probs


def calc_set_diff(docs):
    global tfidf_vec
    source_data = " .".join(docs[1:])
    target_data = docs[0]
    doc_term = tfidf_vec.fit_transform([source_data, target_data])
    source_set = set(np.nonzero(doc_term[0])[1].tolist())
    target_set = set(np.nonzero(doc_term[1])[1].tolist())
    diff = len(target_set - source_set)
    return diff


def calc_geo_diff(docs):
    global tfidf_vec
    source_datas = docs[1:]
    target_data = docs[0]
    doc_term = tfidf_vec.fit_transform(source_datas + [target_data])
    cos = max(
        [
            float(cosine_similarity(doc_term[i], doc_term[-1])[0][0])
            for i in range(len(source_datas))
        ]
    )
    return cos


def calc_kl_div(docs):
    global count_vec
    source_data = " . ".join(docs[1:])
    target_data = docs[0]
    doc_term = count_vec.fit_transform([source_data, target_data])
    source_count = doc_term[0].todense()
    target_count = doc_term[1].todense()
    source_dist = [
        0.5 if source_count[0, i] == 0 else source_count[0, i]
        for i in range(source_count.shape[1])
    ]
    target_dist = [
        0.5 if target_count[0, i] == 0 else target_count[0, i]
        for i in range(target_count.shape[1])
    ]
    kl = entropy(target_dist, source_dist)
    return kl


def calc_tfidf_novelty_score(docs):
    global tfidf_vec1, count_vec
    source_datas = docs[1:]
    target_data = docs[0]
    doc_term = tfidf_vec1.fit_transform(source_datas + [target_data])
    doc_term1 = count_vec.fit_transform(source_datas + [target_data])
    if doc_term1.sum(axis=1)[-1, 0] == 0:
        score = 0.0
    else:
        score = doc_term.sum(axis=1)[-1, 0] / float(doc_term1.sum(axis=1)[-1, 0])
    return score


def calc_pv(docs):
    source_data = " . ".join(docs[1:])
    target_data = docs[0]
    source_vec = model.infer_vector(
        doc_words=[
            token for token in word_tokenize(source_data) if token not in stopwords
        ],
        alpha=0.1,
        min_alpha=0.0001,
        steps=5,
    )
    target_vec = model.infer_vector(
        doc_words=[
            token for token in word_tokenize(target_data) if token not in stopwords
        ],
        alpha=0.1,
        min_alpha=0.0001,
        steps=5,
    )
    return np.concatenate((target_vec, source_vec), axis=0)


def read_dataset(dataset):
    if dataset == "dlnd":
        with open(".data/dlnd/TAP-DLND-1.0_LREC2018_modified/dlnd.jsonl", "r") as f:
            data = f.readlines()
            d = [json.loads(i) for i in data]
            data = []
            for i in d:
                data.append(
                    [i["target_text"]]
                    + [i["source"]]
                    + [1 if i["DLA"] == "Novel" else 0]
                )
        optpath = "dlnd_baselines_class_probs"

    elif dataset == "apwsj":
        with open(".data/apwsj/trec/apwsj.jsonl", "r") as f:
            data = f.readlines()
            d = [json.loads(i) for i in data]
            data = []
            for i in d:
                data.append(
                    [i["target_text"]]
                    + [i["source"]]
                    + [1 if i["DLA"] == "Novel" else 0]
                )
        optpath = "apwsj_baselines_class_probs"

    elif dataset == "webis":
        with open(".data/webis/Webis-CPC-11/webis.jsonl", "r") as f:
            data = f.readlines()
            d = [json.loads(i) for i in data]
            data = []
            for i in d:
                data.append(
                    [i["target_text"]] + [i["source"]] + [1 if i["DLA"] == True else 0]
                )
        optpath = "webis_baselines_class_probs"
    return data, optpath


def plot(labeled_list, acc_vals):

    fig = plt.figure(figsize=(8, 6))
    for model_type, test_acc_list in acc_vals.items():
        plt.plot(labeled_list, test_acc_list)
        plt.title("Varying Labeled Set Size")
        plt.xlabel("Labeled Set Size")
        plt.xscale("log")
        plt.ylabel("Test Accuracy")
    plt.legend(list(acc_vals.keys()))
    new_path = os.path.join("plots", f"vary_labeled.png")
    ver = 0
    if not os.path.exists("plots"):
        os.makedirs("plots")
    while os.path.exists(new_path):
        ver += 1
        new_path = os.path.join("plots", f"vary_labeled{str(ver)}.png")

    fig.savefig(new_path)


# if __name__ == "__main__":
# %%
data, optpath = read_dataset("dlnd")
n_cases = len(data)
labels = [data[i][-1] for i in range(n_cases)]
class_order = np.unique(labels)


labeled_sizes = [
    2,
    4,
    6,
    8,
    20,
    40,
    80,
    100,
    200,
    400,
    600,
    1000,
    1200,
    1400,
    1600,
    2000,
    3000,
]
acc_vals = {}

print("\nSET DIFFERENCE METRIC\n")
logger.info("\nSET DIFFERENCE METRIC\n")
set_diff = [calc_set_diff(data[i][:-1]) for i in range(n_cases)]
vary_acc_set_diff = [100 * i for i in vary_labeled(set_diff)]
acc_vals["Set Difference"] = vary_acc_set_diff


print("\nGEO DIFFERENCE METRIC\n")
logger.info("\nGEO DIFFERENCE METRIC\n")
geo_diff = [calc_geo_diff(data[i][:-1]) for i in range(n_cases)]
vary_acc_geo_diff = [100 * i for i in vary_labeled(geo_diff)]
acc_vals["Geo Difference"] = vary_acc_geo_diff


# print("\nTFIDF NOVELTY SCORE METRIC\n")
# logger.info("\nTFIDF NOVELTY SCORE METRIC\n")
# tfidf_novelty_score = [calc_tfidf_novelty_score(data[i][:-1]) for i in range(n_cases)]
# vary_acc_tfidf_novelty_score = [100 * i for i in vary_labeled(tfidf_novelty_score)]
# acc_vals["TF-IDF"] = vary_acc_tfidf_novelty_score

print("\nKL DIVERGENCE METRIC\n")
logger.info("\nKL DIVERGENCE METRIC\n")
kl_div = [calc_kl_div(data[i][:-1]) for i in range(n_cases)]
vary_acc_kl_div = [100 * i for i in vary_labeled(kl_div)]
acc_vals["KL Div"] = vary_acc_kl_div


# %%
han_acc_vals = [
    50.87396504139834,
    48.02207911683533,
    49.770009199632014,
    52.897884084636615,
    51.057957681692734,
    51.79392824287029,
    57.8656853725851,
    60.71757129714811,
    63.66145354185833,
    75.06899724011039,
    74.60901563937442,
    77.36890524379025,
    81.78472861085557,
    84.360625574977,
    84.4526218951242,
    82.8886844526219,
    86.56853725850966,
]
dan_acc_vals = [
    57.7736890524379,
    56.39374425022999,
    55.38178472861085,
    55.289788408463664,
    50.78196872125115,
    58.233670653173874,
    59.889604415823364,
    57.31370745170193,
    62.83348666053358,
    68.26126954921803,
    73.04507819687213,
    77.46090156393744,
    79.7608095676173,
    81.60073597056117,
    80.86476540938362,
    80.22079116835327,
    85.00459981600736,
]


acc_vals["HAN"] = han_acc_vals
acc_vals["DAN"] = dan_acc_vals

plot(labeled_sizes, acc_vals)


# print("\nPARAGRAPH VECTOR + LR\n")
# logger.info("\nPARAGRAPH VECTOR + LR\n")
# pv = [calc_pv(data[i][:-1]) for i in range(n_cases)]
# probs_pv = train_lr(pv)

# with open(f"./results/novelty_baseline/{optpath}.p", "wb") as f:
#     pickle.dump(
#         {
#             "class_order": class_order,
#             "probs_set_diff": probs_set_diff,
#             "probs_geo_diff": probs_geo_diff,
#             "probs_tfidf_novelty_score": probs_tfidf_novelty_score,
#             "probs_kl_div": probs_kl_div,
#             "probs_pv": probs_pv,
#             "labels": labels,
#         },
#         f,
#     )
# logger.info(
#     f"predictions saved at ./results/novelty_baseline/{optpath}.p"
# )

# %%
# %%


def plot(labeled_list, acc_vals):

    fig = plt.figure(figsize=(8, 6))
    for model_type, test_acc_list in acc_vals.items():
        plt.plot(labeled_list, test_acc_list)
        plt.title("Varying Labeled Set Size")
        plt.xscale("log")
        plt.xlabel("Labeled Set Size")
        plt.ylabel("Test Accuracy")
    plt.legend(list(acc_vals.keys()))
    new_path = os.path.join("plots", f"vary_labeled.png")
    ver = 0
    if not os.path.exists("plots"):
        os.makedirs("plots")
    while os.path.exists(new_path):
        ver += 1
        new_path = os.path.join("plots", f"vary_labeled{str(ver)}.png")

    fig.savefig(new_path)


plot(labeled_sizes, acc_vals)

# %%
