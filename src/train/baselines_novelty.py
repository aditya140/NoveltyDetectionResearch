import os, pickle, logging, shutil, wget, string, json, nltk, time
from random import shuffle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import xml.etree.ElementTree as ET
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from nltk import word_tokenize

import neptune


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
    doc2vec_url = "https://public.boxcloud.com/d/1/b1!lMgOFAdscMZy2rv2GRTRFCr5JTeXMJjXQ0nwcPQoO7ppvzbo4IY7lSF_0oLjW9Zt2CcsI-4JhMKJpUSEJsR59-PgLFlZ68_-ev7AZU9yyXQp7GPy3Hs4i_hQ1X1XSsuv21Z1anQxobdPkrtiQhl-osD7K6Bg4OeF78ouBHp3uz1nsv1HBOndO4vy_miDHuv7j-vlKfZqawHqeSZONpZmTe1oQ1JM8Y3GEbBLQMx2vgajf6qQmuy4KnGObc21iqZR8zoYglpTKPUfYm6ueGFehRlXmQd_6f8_dzIDDar-GFPwKl4YbbvqjKv4smHgORbzuck2UgyO9A7AkIGo03N9qcY7LpWTPnukPPQ1nDLcVZ_Pc2uda1e8QLcgCpS39i5AAWw986HIwOH9AMVXVtI96badck6IG8P7-JSC_ww8QusZdZxUTpMCk_gc4eJD5LBbDUntnMpQ-f85dw-8ANvFJCac_-hrv4Y5xFrs0yvqvkl7jZgx36fpvzyMHqYXbvoVNF5Q8eQzFRncN-H71nq2S4bLdG3LjE00hSR2zIuLzai_i7NdcJZQWU7TGnaloh7kcEI_ekJ1RN85j3HJwmd-x9SH7Ec0ATAbU7_RtuLtc1jSYk5dEIZByilZR2Q99l2YF6tBCPbuOiw55gXzr0CXbgDNJXMCgU3YAybW0QGeo7kcBCIGV6E-Q0IJWp1oyPCyn1TDgh7VuhI4BWuKz6dhSxyCjkM0KggDQNICtkcSA-J68LmABsK1lhcQhQcwoFK5XXhiKbaDb4uLstb37mTUM3UvAlbdz4_JhTrPD45XOoURy569JiNDQTB-QlXkOZP1LbJ26fJtwhGXrEci5xGEcvc-GkurpRpmQGDBPGCPsZ3TVeRgd83RuVCbh_OlPtNLTGEB9qkK_5tm4Ib-8RvmPFch9oQQuc8nu5hjzQS5OIeUkV0INh9XrACI7JCEonaPUgp8bNySjXxMuDh0EphxoRP5CPrE1eh78OEaeGWnuGgLDQbOhPxhVmCd0eSzhIOijQnLh0W9Z7ytGO0gcZ7StjeRUL_1bXxBQo9vYuzQu7mQjOZ68ozG04XgmPEO8-Q0qQJSg0VCSTbJnptABsMdCEet25YbyjbgZ1Hed0M-OSOwsdB1E89DwMU1FXDAt33z1usDmH2S69DRqY1jXyKBCDWCScTQO4xQeHV0vD5f4g3ZwpPwn5kFAjRGbEArUTyHN5D9plKOBBEqTMZJYuaGojZi_d6qaslfMGA9NNh050Mg9Y5kkcOPvsmstqdN-A2BTakdv1i6e_PXkFifWEekLWKIdCeAYMeuhOeauqA6R3e0ntQNGFOQNo09GRJ6ACBEz4QPHfKmYjijRQFVoYGFsItxEK7baEdc8d54/download"
    wget.download(doc2vec_url, "enwiki_dbow.tgz")


def unpack_doc2vecmodel():
    if os.path.exists("enwiki_dbow.tgz"):
        shutil.unpack_archive("./enwiki_dbow.tgz", "enwiki_dbow")


if not os.path.exists("./enwiki_dbow/enwiki_dbow"):
    download_doc2vecmodel()
    unpack_doc2vecmodel()

model = Doc2Vec.load("./enwiki_dbow/enwiki_dbow/doc2vec.bin")
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


def train_lr(features):
    global labels, cv
    class_order = np.unique(labels)
    lr = LogisticRegression(verbose=2, n_jobs=-1)
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


if __name__ == "__main__":

    with open(".data/dlnd/TAP-DLND-1.0_LREC2018_modified/dlnd.jsonl", "r") as f:
        data = f.readlines()
        d = [json.loads(i) for i in data]
        data = []
        for i in d:
            data.append(
                [i["target_text"]] + [i["source"]] + [1 if i["DLA"] == "Novel" else 0]
            )
    n_cases = len(data)
    labels = [data[i][-1] for i in range(n_cases)]
    class_order = np.unique(labels)
    cv = make_cv_10_fold(labels)
    print("\nSET DIFFERENCE METRIC\n")
    logger.info("\nSET DIFFERENCE METRIC\n")
    set_diff = [calc_set_diff(data[i][:-1]) for i in range(n_cases)]
    probs_set_diff = train_lr(set_diff)
    print("\nGEO DIFFERENCE METRIC\n")
    logger.info("\nGEO DIFFERENCE METRIC\n")
    geo_diff = [calc_geo_diff(data[i][:-1]) for i in range(n_cases)]
    probs_geo_diff = train_lr(geo_diff)
    print("\nTFIDF NOVELTY SCORE METRIC\n")
    logger.info("\nTFIDF NOVELTY SCORE METRIC\n")
    tfidf_novelty_score = [
        calc_tfidf_novelty_score(data[i][:-1]) for i in range(n_cases)
    ]
    probs_tfidf_novelty_score = train_lr(tfidf_novelty_score)
    print("\nKL DIVERGENCE METRIC\n")
    logger.info("\nKL DIVERGENCE METRIC\n")
    kl_div = [calc_kl_div(data[i][:-1]) for i in range(n_cases)]
    probs_kl_div = train_lr(kl_div)
    print("\nPARAGRAPH VECTOR + LR\n")
    logger.info("\nPARAGRAPH VECTOR + LR\n")
    pv = [calc_pv(data[i][:-1]) for i in range(n_cases)]
    probs_pv = train_lr(pv)

    with open("./results/novelty_baseline/dlnd_baselines_class_probs.p", "wb") as f:
        pickle.dump(
            {
                "class_order": class_order,
                "probs_set_diff": probs_set_diff,
                "probs_geo_diff": probs_geo_diff,
                "probs_tfidf_novelty_score": probs_tfidf_novelty_score,
                "probs_kl_div": probs_kl_div,
                "probs_pv": probs_pv,
                "labels": labels,
            },
            f,
        )
    logger.info(
        "predictions saved at ./results/novelty_baseline/dlnd_baselines_class_probs.p"
    )
