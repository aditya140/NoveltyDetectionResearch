import sys, copy

sys.path.append(".")
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import datetime
import time
import shutil
import neptune
from millify import millify
import matplotlib.pyplot as plt
import pickle

from src.defaults import *
from src.model.novelty_models import *
from src.datasets.novelty import *
from src.model.nli_models import *
from src.utils.trainer import *
from src.train.train_novelty import Train_novelty

labeled_size = [
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


if __name__ == "__main__":
    args = parse_novelty_conf()
    dataset_conf, optim_conf, model_type, model_conf, sentence_field = get_novelty_conf(
        args
    )
    test_acc_list = []
    labeled_list = []
    for labeled in labeled_size:
        try:
            (
                args_c,
                dataset_conf_c,
                model_conf_c,
                optim_conf_c,
                model_type_c,
                sentence_field_c,
            ) = (
                copy.deepcopy(args),
                copy.deepcopy(dataset_conf),
                copy.deepcopy(model_conf),
                copy.deepcopy(optim_conf),
                copy.deepcopy(model_type),
                copy.deepcopy(sentence_field),
            )

            dataset_conf_c["labeled"] = labeled
            trainer = Train_novelty(
                args_c,
                dataset_conf_c,
                model_conf_c,
                optim_conf_c,
                model_type_c,
                sentence_field_c,
            )
            test_acc = trainer.fit(
                **{"batch_attr": {"model_inp": ["source", "target"], "label": "label"}}
            )
            test_acc_list.append(test_acc)
            labeled_list.append(labeled)
        except:
            print(f"Failed for labeled Size {labeled}")

    neptune.init(
        project_qualified_name=VARY_LABELED,
        api_token=NEPTUNE_API,
    )

    exp = neptune.create_experiment()
    exp_id = exp.id
    neptune.log_text("Dataset Conf", str(dataset_conf))
    neptune.log_text("Model Conf", str(model_conf))
    neptune.log_text("Hparams", str(optim_conf))
    neptune.append_tag([dataset_conf["dataset"], model_type])

    print(labeled_list)
    print(test_acc_list)
    print(model_type)

    acc_vals = {
        "labeled_list": labeled_list,
        "test_acc_list": test_acc_list,
        "model_type": model_type,
    }

    fig = plt.figure()
    plt.plot(labeled_list, test_acc_list)
    plt.title("Varying Labeled Set Size")
    plt.xlabel("Labeled Set Size")
    plt.ylabel("Test Accuracy")
    plt.legend([model_type])
    new_path = os.path.join("plots", f"vary_labeled_{model_type}.png")
    ver = 0
    if not os.path.exists("plots"):
        os.makedirs("plots")
    while os.path.exists(new_path):
        ver += 1
        new_path = os.path.join("plots", f"vary_labeled_{model_type}{str(ver)}.png")
        new_path_pickle = os.path.join(
            "plots", f"vary_labeled_{model_type}{str(ver)}.p"
        )

    with open(new_path_pickle, "wb") as handle:
        pickle.dump(acc_vals, handle, protocol=pickle.HIGHEST_PROTOCOL)

    neptune.log_image("vary_labeled_size", fig, image_name="vary_labeled_size")
    neptune.log_test("labeled_list", ",".join([str(i) for i in labeled_list]))
    neptune.log_test("test_acc_list", ",".join([str(i) for i in test_acc_list]))
    fig.savefig(new_path)

    neptune.log_artifact(new_path)
