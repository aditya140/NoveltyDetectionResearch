### Reference to the the root project directory
import sys

sys.path.append(".")
ROOT_PATH = "./"


# Imports
import process_apwsj
import nltk
import requests, os
import svn.remote
import argparse
import shutil
from os.path import join as osj
from utils.drive_utils import download_file_from_google_drive


nltk.download("punkt")


def process_glove():
    from gensim.scripts.glove2word2vec import glove2word2vec

    glove2word2vec(
        glove_input_file=osj(ROOT_PATH, "glove.840B.300d.txt"),
        word2vec_output_file=osj(ROOT_PATH, "glove_vec.txt"),
    )
    from gensim.models.keyedvectors import KeyedVectors
    import pickle

    glove_model = KeyedVectors.load_word2vec_format(
        osj(ROOT_PATH, "glove_vec.txt"), binary=False
    )
    with open(osj(ROOT_PATH, "glove.pkl"), "wb") as f:
        pickle.dump(glove_model, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Novelty DAN Training")

    parser.add_argument("--glove", type=int, help="process glove")
    parser.add_argument("--document", type=int, help="process document data")
    parser.add_argument("--novelty", type=int, help="process novelty data")
    parser.add_argument("--snli", type=int, help="process snli data")

    args = parser.parse_args()

    print("Working dir:", os.getcwd())

    if args.novelty == 1:
        download_file_from_google_drive(
            "1q-P3ReGf-yWnKrhb6XQAuMGo39hXlhYG", osj(ROOT_PATH, "dlnd.zip")
        )

        download_file_from_google_drive(
            "1h7bS3zdP-6bPDuvJ_JXq8YzR6a3OEmRg", osj(ROOT_PATH, "dataset_apw.zip")
        )

        
        # Unpack all zip files (datasets)
        shutil.unpack_archive(
            osj(ROOT_PATH, "CMUNRF1.tar"), osj(ROOT_PATH, "./dataset/apwsj/")
        )

        shutil.unpack_archive(
            osj(ROOT_PATH, "dataset_apw.zip"), osj(ROOT_PATH, "./dataset/")
        )

        shutil.unpack_archive(
            osj("./dataset/trec", "AP.tar"), osj("./dataset/trec","AP")
        )
        shutil.unpack_archive(
            osj("./dataset/trec", "trec.zip"), osj("./dataset/trec","trec")
        )
        shutil.unpack_archive(
            osj("./dataset/trec", "wsj_split.zip"), osj("./dataset/trec","WSJ")
        )

        os.remove(osj("./dataset/trec", "AP.tar"))
        os.remove(osj("./dataset/trec", "wsj_split.zip"))
        os.remove(osj("./dataset/trec", "trec.zip"))


        shutil.unpack_archive(
            osj(ROOT_PATH, "Webis-CPC-11.zip"),
            osj(ROOT_PATH, "./dataset/novelty/webis/"),
        )

        shutil.unpack_archive(
            osj(ROOT_PATH, "dlnd.zip"), osj(ROOT_PATH, "./dataset/novelty/dlnd/")
        )
        # download TREC data
        r = svn.remote.RemoteClient(
            "http://svn.dridan.com/sandpit/QA/trecdata/datacollection/"
        )
        r.checkout(osj(ROOT_PATH, "./dataset/trec"))
        process_apwsj.create_json()

    if args.document == 1:
        shutil.unpack_archive(
            osj(ROOT_PATH, "yelp-dataset.zip"), osj(ROOT_PATH, "./dataset/yelp/")
        )

        shutil.unpack_archive(
            osj(ROOT_PATH, "arxiv.zip"), osj(ROOT_PATH, "./dataset/arxiv/")
        )
        shutil.unpack_archive(
            osj(ROOT_PATH, "reuters21578.tar.gz"), osj(ROOT_PATH, "./dataset/reuters/")
        )

    if args.snli == 1:
        shutil.unpack_archive(
            osj(ROOT_PATH, "snli_1.0.zip"), osj(ROOT_PATH, "./dataset/snli/")
        )

    if args.glove == "1":
        process_glove()
