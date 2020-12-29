### Reference to the the root project directory
import sys

sys.path.append(".")
ROOT_PATH = "./"

import shutil
import os
import neptune
from utils.keys import NEPTUNE_API
import argparse
from os.path import join as osj

# ROOT_PATH = "../"


bilstm_enc = "bilstm_encoder.zip"
attn_enc = "attn_encoder.zip"
cnn_novelty = "cnn_novelty.zip"
document_imdb_han_clf = "document_imdb_han_clf.zip"
document_imdb_han_reg = "document_imdb_han_reg.zip"


neptune_experiments = {
    # "attn_enc": "SNLI-105",
    "attn_enc": "SNLI-411",
    "bilstm_enc": "SNLI-104",
    "document_imdb_han_clf": "IM-33",
    "document_imdb_han_reg": "IM-34",
    "cnn_novelty": "NOVELTY-315",
}


def make_paths():

    """Create Model folders in the models directory of the project
    """
    if not os.path.exists("./models/bilstm_encoder"):
        os.makedirs("./models/bilstm_encoder")

    if not os.path.exists("./models/attn_encoder"):
        os.makedirs("./models/attn_encoder")

    if not os.path.exists("./models/cnn_novelty"):
        os.makedirs("./models/cnn_novelty")

    if not os.path.exists("./models/document_imdb_han_clf"):
        os.makedirs("./models/document_imdb_han_clf")

    if not os.path.exists("./models/document_imdb_han_reg"):
        os.makedirs("./models/document_imdb_han_reg")


def extract_zips():
    """Extract the downloaded zip files into the respective model folders
    """
    shutil.unpack_archive(
        osj(ROOT_PATH, "./models", bilstm_enc),
        extract_dir=osj(ROOT_PATH, "./models/bilstm_encoder"),
    )
    shutil.unpack_archive(
        osj(ROOT_PATH, "./models", attn_enc),
        extract_dir=osj(ROOT_PATH, "./models/attn_encoder"),
    )
    shutil.unpack_archive(
        osj(ROOT_PATH, "./models", cnn_novelty),
        extract_dir=osj(ROOT_PATH, "./models/cnn_novelty"),
    )
    shutil.unpack_archive(
        osj(ROOT_PATH, "./models", document_imdb_han_clf),
        extract_dir=osj(ROOT_PATH, "./models/document_imdb_han_clf"),
    )
    shutil.unpack_archive(
        osj(ROOT_PATH, "./models", document_imdb_han_reg),
        extract_dir=osj(ROOT_PATH, "./models/document_imdb_han_reg"),
    )


def download_from_neptune():

    make_paths()

    project = neptune.init("aparkhi/SNLI", api_token=NEPTUNE_API)
    experiment = project.get_experiments(id=neptune_experiments["bilstm_enc"])[0]
    experiment.download_artifact(bilstm_enc, osj(ROOT_PATH, "./models"))

    project = neptune.init("aparkhi/SNLI", api_token=NEPTUNE_API)
    experiment = project.get_experiments(id=neptune_experiments["attn_enc"])[0]
    experiment.download_artifact(attn_enc, osj(ROOT_PATH, "./models"))

    project = neptune.init("aparkhi/DocClassification", api_token=NEPTUNE_API)
    experiment = project.get_experiments(
        id=neptune_experiments["document_imdb_han_clf"]
    )[0]
    experiment.download_artifact(document_imdb_han_clf, osj(ROOT_PATH, "./models"))

    project = neptune.init("aparkhi/DocClassification", api_token=NEPTUNE_API)
    experiment = project.get_experiments(
        id=neptune_experiments["document_imdb_han_reg"]
    )[0]
    experiment.download_artifact(document_imdb_han_reg, osj(ROOT_PATH, "./models"))

    project = neptune.init("aparkhi/Novelty", api_token=NEPTUNE_API)
    experiment = project.get_experiments(id=neptune_experiments["cnn_novelty"])[0]
    experiment.download_artifact(cnn_novelty, osj(ROOT_PATH, "./models"))

    extract_zips()


# Depricated : Use download from neptune
def download_from_drive():
    Model_path = "/content/drive/My Drive/Thesis/Novelty Goes Deep"
    bilstm_enc = "bilstm_encoder.zip"
    attn_enc = "attn_encoder.zip"
    cnn_novelty = "cnn_novelty.zip"
    document_imdb_han_clf = "document_imdb_han_clf.zip"
    document_imdb_han_reg = "document_imdb_han_reg.zip"
    make_paths()
    shutil.copyfile(
        os.path.join(Model_path, bilstm_enc), os.path.join("./models", bilstm_enc)
    )
    shutil.copyfile(
        os.path.join(Model_path, attn_enc), os.path.join("./models", attn_enc)
    )
    shutil.copyfile(
        os.path.join(Model_path, cnn_novelty), os.path.join("./models", cnn_novelty)
    )
    shutil.copyfile(
        os.path.join(Model_path, document_imdb_han_clf),
        os.path.join("./models", document_imdb_han_clf),
    )
    shutil.copyfile(
        os.path.join(Model_path, document_imdb_han_reg),
        os.path.join("./models", document_imdb_han_reg),
    )
    extract_zips()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Models")
    parser.add_argument(
        "--neptune", action="store_true", help="Download saved model", default=False
    )
    # parser.add_argument(
    #     "--drive", action="store_true", help="Save model", default=False
    # )
    args = parser.parse_args()
    print("Downloading saved models from Neptune")
    if args.neptune:
        download_from_neptune()
    elif args.drive:
        download_from_drive()