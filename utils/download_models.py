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


neptune_experiments_to_download = {
    ("SNLI", "attn_encoder"): ["SNLI-12"],
    ("SNLI", "bilstm_encoder"): ["SNLI-13"],
    ("DocClassification", "han"): ["DOC-2",'DOC-4'],
}


def download_model(project, model_type, _id):
    model_folder_path = "./models"
    model_path = osj(model_folder_path, model_type, _id)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    project = neptune.init(f"aparkhi/{project}", api_token=NEPTUNE_API)
    experiment = project.get_experiments(id=_id)[0]
    experiment.download_artifact(_id + ".zip", osj(model_folder_path, model_type))

    shutil.unpack_archive(
        osj(model_folder_path, model_type, _id + ".zip"),
        extract_dir=model_path,
    )


def download_from_neptune():
    for project, ids in neptune_experiments_to_download.items():
        for _id in ids:
            download_model(project[0], project[1], _id)


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