from argparse import ArgumentParser
import os
import random
import numpy as np
import torch
import logging
import time
import dill
import neptune
import hyperdash
import shutil
import dill
import pickle
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

NEPTUNE_API = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTg3MzU5NjQtMmIxZC00Njg0LTgzYzMtN2UwYjVlYzVhNDg5In0="
NLI_NEPTUNE_PROJECT = "aparkhi/NLI"
NOVELTY_NEPTUNE_PROJECT = "aparkhi/Novelty"
NOVELTY_ENSEMBLE_NEPTUNE_PROJECT = "aparkhi/NoveltyEnsemble"


"""
NLI Argument Parser
"""


def parse_nli_conf():
    parser = ArgumentParser(description="PyTorch/torchtext NLI Baseline")
    parser.add_argument("--dataset", "-d", type=str, default="mnli")

    # language
    parser.add_argument("--tokenizer", type=str, default="bert")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_char_emb", type=bool, default=False)
    parser.add_argument("--folds", type=bool, default=False)
    parser.add_argument("--max_word_len", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    # optimizer_conf
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--loss_agg", type=str, default="sum")
    parser.add_argument("--scheduler", type=str, default="step")

    subparsers = parser.add_subparsers(dest="model_type")

    # model_conf
    parser_attention = subparsers.add_parser("attention")
    attention_model_params(parser_attention)

    # model_conf
    parser_bilstm = subparsers.add_parser("bilstm")
    bilstm_model_params(parser_bilstm)

    # model_conf
    parser_struc_attn = subparsers.add_parser("struc_attn")
    struc_attn_model_params(parser_struc_attn)

    # model_conf
    parser_mwan = subparsers.add_parser("mwan")
    mwan_model_params(parser_mwan)

    parser.add_argument("--results_dir", type=str, default="results")
    return check_args(parser.parse_args())


def get_nli_conf(args):
    # hparams
    hparams = {}
    hparams["optimizer"] = {
        "optim": args.optim,
        "lr": args.lr,
        "scheduler": args.scheduler,
    }
    hparams["epochs"] = args.epochs
    hparams["loss_agg"] = args.loss_agg

    # dataset config
    dataset_conf = {}
    dataset_conf["dataset"] = args.dataset
    dataset_conf["tokenizer"] = args.tokenizer
    dataset_conf["use_char_emb"] = args.use_char_emb
    dataset_conf["max_len"] = args.max_len
    dataset_conf["batch_size"] = args.batch_size
    dataset_conf["device"] = args.device
    dataset_conf["max_word_len"] = args.max_word_len

    used_keys = [
        "tokenizer",
        "max_len",
        "batch_size",
        "epochs",
        "lr",
        "optim",
        "loss_agg",
        "scheduler",
        "model_type",
    ]

    model_type = args.model_type
    model_conf = {
        k: args.__dict__[k] for k in set(list(args.__dict__.keys())) - set(used_keys)
    }
    return dataset_conf, hparams, model_type, model_conf


def parse_nli_pl_conf():
    parser = ArgumentParser(description="PyTorch/torchtext NLI Baseline")
    parser.add_argument("--dataset", "-d", type=str, default="mnli")

    # language
    parser.add_argument("--tokenizer", type=str, default="bert")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--use_char_emb", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")

    # optimizer_conf
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr_base", type=float, default=0.001)
    parser.add_argument("--lr_tune", type=float, default=0.001)
    parser.add_argument("--optim_base", type=str, default="adamw")
    parser.add_argument("--optim_tune", type=str, default="adam")
    parser.add_argument("--switch_epoch", type=int, default=5)
    parser.add_argument("--loss_agg", type=str, default="sum")
    parser.add_argument("--scheduler", type=str, default="step")

    subparsers = parser.add_subparsers(dest="model_type")

    # model_conf
    parser_attention = subparsers.add_parser("attention")
    attention_model_params(parser_attention)

    # model_conf
    parser_bilstm = subparsers.add_parser("bilstm")
    bilstm_model_params(parser_bilstm)

    # model_conf
    parser_struc_attn = subparsers.add_parser("struc_attn")
    struc_attn_model_params(parser_struc_attn)

    parser.add_argument("--results_dir", type=str, default="results")
    return check_args(parser.parse_args())


def get_nli_conf_pl(args):
    # hparams
    hparams = {}
    hparams["optimizer_base"] = {
        "optim": args.optim_base,
        "lr": args.lr_base,
        "scheduler": args.scheduler,
    }
    hparams["optimizer_tune"] = {
        "optim": args.optim_tune,
        "lr": args.lr_tune,
        "scheduler": args.scheduler,
    }
    hparams["switch_epoch"] = args.switch_epoch
    hparams["epochs"] = args.epochs
    hparams["loss_agg"] = args.loss_agg

    # dataset config
    dataset_conf = {}
    dataset_conf["dataset"] = args.dataset
    dataset_conf["tokenizer"] = args.tokenizer
    dataset_conf["max_len"] = args.max_len
    dataset_conf["batch_size"] = args.batch_size
    dataset_cong["use_char_emb"] = args.use_char_emb
    dataset_conf["device"] = args.device

    used_keys = [
        "tokenizer",
        "max_len",
        "batch_size",
        "epochs",
        "lr_base",
        "lr_tune",
        "optim_base",
        "optim_tune",
        "switch_epoch",
        "loss_agg",
        "scheduler",
        "model_type",
    ]

    model_type = args.model_type
    model_conf = {
        k: args.__dict__[k] for k in set(list(args.__dict__.keys())) - set(used_keys)
    }
    return dataset_conf, hparams, model_type, model_conf


"""
NLI Model Configurations
"""


def attention_model_params(parser_dump):
    parser_dump.add_argument("--hidden_size", type=int, default=400)
    parser_dump.add_argument("--embedding_dim", type=int, default=300)
    parser_dump.add_argument("--char_embedding_dim", type=int, default=100)
    parser_dump.add_argument("--dropout", type=float, default=0.3)
    parser_dump.add_argument("--use_glove", type=bool, default=False)
    parser_dump.add_argument("--num_layers", type=int, default=1)
    parser_dump.add_argument("--fcs", type=int, default=1)
    parser_dump.add_argument("--attention_layer_param", type=int, default=200)


def bilstm_model_params(parser_dump):
    parser_dump.add_argument("--hidden_size", type=int, default=400)
    parser_dump.add_argument("--embedding_dim", type=int, default=300)
    parser_dump.add_argument("--char_embedding_dim", type=int, default=100)
    parser_dump.add_argument("--dropout", type=float, default=0.3)
    parser_dump.add_argument("--use_glove", type=bool, default=False)
    parser_dump.add_argument("--num_layers", type=int, default=1)


def struc_attn_model_params(parser_dump):
    parser_dump.add_argument("--hidden_size", type=int, default=400)
    parser_dump.add_argument("--embedding_dim", type=int, default=300)
    parser_dump.add_argument("--char_embedding_dim", type=int, default=100)
    parser_dump.add_argument("--dropout", type=float, default=0.3)
    parser_dump.add_argument("--use_glove", type=bool, default=False)
    parser_dump.add_argument("--num_layers", type=int, default=1)
    parser_dump.add_argument("--fcs", type=int, default=1)
    parser_dump.add_argument("--r", type=int, default=5)
    parser_dump.add_argument("--attention_layer_param", type=int, default=200)
    parser_dump.add_argument("--gated_embedding_dim", type=int, default=150)
    parser_dump.add_argument("--gated", type=bool, default=False)
    parser_dump.add_argument("--pool_strategy", type=str, default="max")


def mwan_model_params(parser_dump):
    parser_dump.add_argument("--hidden_size", type=int, default=150)
    parser_dump.add_argument("--embedding_dim", type=int, default=300)
    parser_dump.add_argument("--char_embedding_dim", type=int, default=100)
    parser_dump.add_argument("--dropout", type=float, default=0.2)
    parser_dump.add_argument("--use_glove", type=bool, default=False)
    parser_dump.add_argument("--freeze_emb", type=bool, default=False)


"""
Novelty Argument Parser
"""


def parse_novelty_conf():
    parser = ArgumentParser(description="PyTorch/torchtext Novelty Training")
    parser.add_argument("--dataset", "-d", type=str, default="dlnd")

    # language
    parser.add_argument("--load_nli", type=str, default="None")
    parser.add_argument("--max_num_sent", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--folds", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tokenizer", type=str, default="None")
    parser.add_argument("--max_len", type=int, default=0)
    parser.add_argument("--sent_tokenizer", type=str, default="spacy")

    # optimizer_conf
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--loss_agg", type=str, default="sum")
    parser.add_argument("--scheduler", type=str, default="step")
    parser.add_argument("--freeze_encoder", type=bool, default=False)

    subparsers = parser.add_subparsers(dest="model_type")

    # model_conf
    parser_dan = subparsers.add_parser("dan")
    dan_model_parameters(parser_dan)

    # model_conf
    parser_adin = subparsers.add_parser("adin")
    adin_model_parameters(parser_adin)

    # model_conf
    parser_han = subparsers.add_parser("han")
    han_model_parameters(parser_han)

    # model_conf
    parser_rdv = subparsers.add_parser("rdv_cnn")
    rdv_cnn_model_parameters(parser_rdv)

    # model_conf
    parser_diin = subparsers.add_parser("diin")
    diin_model_parameters(parser_diin)

    # model_conf
    parser_diin = subparsers.add_parser("mwan")
    mwan_nov_model_parameters(parser_diin)

    # model_conf
    parser_stru = subparsers.add_parser("struc")
    struc_self_attn_model_parameters(parser_stru)

    parser.add_argument("--results_dir", type=str, default="results")
    return check_args(parser.parse_args())


def get_novelty_conf(args):
    # hparams
    hparams = {}
    hparams["optimizer"] = {
        "optim": args.optim,
        "lr": args.lr,
        "scheduler": args.scheduler,
    }
    hparams["epochs"] = args.epochs
    hparams["loss_agg"] = args.loss_agg

    # dataset config
    dataset_conf = {}
    dataset_conf["dataset"] = args.dataset
    dataset_conf["max_num_sent"] = args.max_num_sent
    dataset_conf["sent_tokenizer"] = args.sent_tokenizer
    dataset_conf["batch_size"] = args.batch_size
    dataset_conf["device"] = args.device

    if args.load_nli == "None":
        assert args.tokenizer != "None"
        assert args.max_len != 0

        dataset_conf["tokenizer"] = args.tokenizer
        dataset_conf["max_len"] = args.max_len
        sentence_field = None

    else:
        check_model(args.load_nli)
        sentence_field = load_field(args.load_nli)

    used_keys = [
        "tokenizer",
        "max_len",
        "batch_size",
        "epochs",
        "lr",
        "optim",
        "loss_agg",
        "scheduler",
        "model_type",
    ]

    model_type = args.model_type
    model_conf = {
        k: args.__dict__[k] for k in set(list(args.__dict__.keys())) - set(used_keys)
    }
    return dataset_conf, hparams, model_type, model_conf, sentence_field


"""
Novelty Detection Model Configurations
"""


def dan_model_parameters(parser_dump):
    parser_dump.add_argument("--hidden_size", type=int, default=400)
    parser_dump.add_argument("--dropout", type=float, default=0.3)


def han_model_parameters(parser_dump):
    parser_dump.add_argument("--hidden_size", type=int, default=400)
    parser_dump.add_argument("--dropout", type=float, default=0.3)
    parser_dump.add_argument("--num_layers", type=int, default=1)
    parser_dump.add_argument("--attention_layer_param", type=int, default=200)


def adin_model_parameters(parser_dump):
    parser_dump.add_argument("--hidden_size", type=int, default=400)
    parser_dump.add_argument("--dropout", type=float, default=0.3)
    parser_dump.add_argument("--k", type=int, default=200)
    parser_dump.add_argument("--num_layers", type=int, default=1)
    parser_dump.add_argument("--N", type=int, default=1)


def rdv_cnn_model_parameters(parser_dump):
    parser_dump.add_argument("--dropout", type=float, default=0.3)
    parser_dump.add_argument("--num_filters", type=int, default=95)
    parser_dump.add_argument("--filter_sizes", type=int, nargs="+", default=[3, 5, 6])


def diin_model_parameters(parser_dump):
    parser_dump.add_argument("--hidden_size", type=int, default=400)
    parser_dump.add_argument(
        "--dropout", type=float, nargs="+", default=[0.3, 0.3, 0.3, 0.3, 0.3]
    )
    parser_dump.add_argument(
        "--dense_net_first_scale_down_ratio", type=float, default=0.3
    )
    parser_dump.add_argument("--dense_net_channels", type=int, default=100)
    parser_dump.add_argument("--dense_net_kernel_size", type=int, default=3)
    parser_dump.add_argument("--dense_net_transition_rate", type=float, default=0.2)
    parser_dump.add_argument("--first_scale_down_kernel", type=int, default=1)
    parser_dump.add_argument("--num_layers", type=int, default=2)
    parser_dump.add_argument("--dense_net_layers", type=int, default=3)
    parser_dump.add_argument("--dense_net_growth_rate", type=int, default=20)


def mwan_nov_model_parameters(parser_dump):
    parser_dump.add_argument("--hidden_size", type=int, default=150)
    parser_dump.add_argument("--dropout", type=float, default=0.3)


def struc_self_attn_model_parameters(parser_dump):
    parser_dump.add_argument("--dropout", type=float, default=0.3)
    parser_dump.add_argument("--hidden_size", type=int, default=300)
    parser_dump.add_argument("--attention_hops", type=int, default=25)
    parser_dump.add_argument("--attention_layer_param", type=int, default=150)
    parser_dump.add_argument("--num_layers", type=int, default=1)
    parser_dump.add_argument("--prune_p", type=int, default=20)
    parser_dump.add_argument("--prune_q", type=int, default=10)


"""
Utils
"""


def check_args(args):
    check_folder(os.path.join(args.results_dir))
    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print("number of epochs must be larger than or equal to one")

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print("batch size must be larger than or equal to one")

    # --seed
    if args.seed != -1:
        seed_torch(args.seed)

    return args


def load_field(_id, field_type="text"):
    results_path = "./results"
    if field_type == "text":
        with open(os.path.join(results_path, _id, "text_field"), "rb") as f:
            field = dill.load(f)

    return field


def check_model(_id):
    model_path = "./results"
    if os.path.exists(os.path.join(model_path, _id)):
        return
    else:
        print(f"Downloading {_id} from neptune")
        download_models_from_neptune(_id)
        return


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def seed_torch(seed=1029):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # seed all gpus
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def get_logger(args, phase, expt_id):
    check_folder(os.path.join(args.results_dir, expt_id))
    logging.basicConfig(
        level=logging.INFO,
        filename="{}/{}/{}_{}.log".format(
            args.results_dir,
            expt_id,
            phase,
            time.strftime("%H:%M:%S", time.gmtime(time.time())),
        ),
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    return logging.getLogger(phase)


def get_device(gpu_no):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_no)
        return torch.device("cuda:{}".format(gpu_no))
    else:
        return torch.device("cpu")


def makedirs(name):
    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def get_vocabs(dataset):
    text_field = dataset.TEXT
    if dataset.options["use_char_emb"]:
        char_field = dataset.CHAR_TEXT
    else:
        char_field = None
    return (text_field, char_field)


def save_field(path, field):
    with open(path, "wb") as f:
        dill.dump(field, f)


def download_models_from_neptune(_id):
    if _id.split("-")[0] == "NLI":
        download_model(NLI_NEPTUNE_PROJECT, _id)


def download_model(project, _id):
    model_folder_path = "./results"
    model_path = os.path.join(model_folder_path, _id)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    project = neptune.init(project, api_token=NEPTUNE_API)
    experiment = project.get_experiments(id=_id)[0]
    experiment.download_artifact(_id + ".zip", model_folder_path)

    shutil.unpack_archive(
        os.path.join(model_folder_path, _id + ".zip"),
        extract_dir=model_path,
    )


def load_encoder_data(_id):
    model_path = os.path.join("./results/", _id, "model.pt")
    model_data = torch.load(model_path)
    return model_data


def get_hyperdash_api():
    return "6dqfQAL9Xij4kBZzoFO+iDTxNHszbaxsxhzaeg0f/DE="


def setup_prc_plot(title):
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    return plt


def plot_prc(plt,probs,gold,cls_label=0,label=""):
    if cls_label==1:
        invert_gold = [1-i for i in gold]
        gold = invert_gold
    p_,r_,_ = precision_recall_curve(gold,[i[cls_label] for i in probs])
    plt.plot(r_,p_,"-",label=label)
    plt.legend(loc='best')
    return plt

"""
Tuning
"""


def parse_novelty_tune_conf():
    parser = ArgumentParser(description="PyTorch/torchtext Novelty Training")
    parser.add_argument("--dataset", "-d", type=str, default="dlnd")

    # language
    parser.add_argument("--load_nli", type=str, default="None")
    parser.add_argument("--max_num_sent", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--folds", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tokenizer", type=str, default="None")
    parser.add_argument("--max_len", type=int, default=0)
    parser.add_argument("--sent_tokenizer", type=str, default="spacy")

    # optimizer_conf
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--sampler", type=str, default="tpe")

    subparsers = parser.add_subparsers(dest="model_type")

    # model_conf
    parser_dan = subparsers.add_parser("dan")
    dan_model_parameters(parser_dan)

    # model_conf
    parser_adin = subparsers.add_parser("adin")
    adin_model_parameters(parser_adin)

    # model_conf
    parser_han = subparsers.add_parser("han")
    han_model_parameters(parser_han)

    # model_conf
    parser_rdv = subparsers.add_parser("rdv_cnn")
    rdv_cnn_model_parameters(parser_rdv)

    # model_conf
    parser_diin = subparsers.add_parser("diin")
    diin_model_parameters(parser_diin)

    # model_conf
    parser_diin = subparsers.add_parser("mwan")
    mwan_nov_model_parameters(parser_diin)

    # model_conf
    parser_stru = subparsers.add_parser("struc")
    struc_self_attn_model_parameters(parser_stru)

    parser.add_argument("--results_dir", type=str, default="results")
    return check_args(parser.parse_args())


def get_tuning_novelty_conf(args):
    # hparams
    hparams = {}
    hparams["optimizer"] = {
        "optim": args.optim,
        "lr": args.lr,
    }
    hparams["epochs"] = args.epochs

    # dataset config
    dataset_conf = {}
    dataset_conf["dataset"] = args.dataset
    dataset_conf["max_num_sent"] = args.max_num_sent
    dataset_conf["sent_tokenizer"] = args.sent_tokenizer
    dataset_conf["batch_size"] = args.batch_size
    dataset_conf["device"] = args.device

    if args.load_nli == "None":
        assert args.tokenizer != "None"
        assert args.max_len != 0

        dataset_conf["tokenizer"] = args.tokenizer
        dataset_conf["max_len"] = args.max_len
        sentence_field = None

    else:
        check_model(args.load_nli)
        sentence_field = load_field(args.load_nli)

    used_keys = [
        "tokenizer",
        "max_len",
        "batch_size",
        "epochs",
        "lr",
        "optim",
        "model_type",
    ]

    model_type = args.model_type
    model_conf = {
        k: args.__dict__[k] for k in set(list(args.__dict__.keys())) - set(used_keys)
    }
    return dataset_conf, hparams, model_type, model_conf, sentence_field


def model_conf_tuning(trial, model_conf, model_type):
    if model_type == "dan":
        return dan_model_tuning(trial, model_conf)
    if model_type == "han":
        return han_model_tuning(trial, model_conf)
    if model_type == "adin":
        return adin_model_tuning(trial, model_conf)
    if model_type == "mwan":
        return mwan_model_tuning(trial, model_conf)
    if model_type == "struc":
        return struc_self_attn_tuning(trial, model_conf)


def dan_model_tuning(trial, model_conf):
    model_conf["hidden_size"] = trial.suggest_categorical(
        "hidden_size", [50, 100, 200, 300, 400]
    )
    return model_conf


def adin_model_tuning(trial, model_conf):
    model_conf["hidden_size"] = trial.suggest_int("hidden_size", 50, 400)
    model_conf["k"] = trial.suggest_int("k", 10, 300)
    model_conf["N"] = trial.suggest_int("N", 1, 3)
    return model_conf


def han_model_tuning(trial, model_conf):
    model_conf["hidden_size"] = trial.suggest_int("hidden_size", 50, 400)
    model_conf["num_layers"] = trial.suggest_int("num_layers", 1, 3)
    model_conf["attention_layer_param"] = trial.suggest_int(
        "attention_layer_param", 10, 300
    )
    return model_conf


def mwan_model_tuning(trial, model_conf):
    model_conf["hidden_size"] = trial.suggest_categorical(
        "hidden_size", [10, 50, 100, 200, 300, 400]
    )
    return model_conf


def struc_self_attn_tuning(trial, model_conf):
    model_conf["hidden_size"] = trial.suggest_int("hidden_size", 50, 400)
    model_conf["num_layers"] = trial.suggest_int("num_layers", 1, 3)
    model_conf["attention_hops"] = trial.suggest_int("attention_hops", 2, 25)
    model_conf["prune_p"] = trial.suggest_int("prune_p", 10, 150)
    model_conf["prune_q"] = trial.suggest_int("prune_q", 5, 100)
    model_conf["attention_layer_param"] = trial.suggest_int(
        "attention_layer_param", 10, 300
    )
    return model_conf


def model_search_space(model_type):
    if model_type == "dan":
        return dan_search_space()
    if model_type == "han":
        return han_search_space()
    if model_type == "adin":
        return adin_search_space()
    if model_type == "mwan":
        return mwan_search_space()
    if model_type == "struc":
        return struc_search_space()


def dan_search_space():
    return {"hidden_size": [50, 100, 200, 300, 400]}


def han_search_space():
    return {
        "hidden_size": [50, 100, 200, 300, 400],
        "num_layers": [1, 2],
        "attention_layer_param": [10, 50, 100, 200, 300, 400],
    }


def adin_search_space():
    return {
        "hidden_size": [50, 100, 200, 300, 400],
        "k": [10, 50, 100, 200, 300],
        "N": [1, 2],
    }


def mwan_search_space():
    return {"hidden_size": [10, 50, 100, 200, 300, 400]}


def struc_search_space():
    return {
        "hidden_size": [50, 100, 200, 300, 400],
        "num_layers": [1, 2],
        "attention_layer_param": [10, 50, 100, 200, 300, 400],
        "attention_hops": [3, 5, 10, 20, 30],
        "prune_p": [10, 50, 100, 150],
        "prune_q": [5, 10, 20, 50, 100],
    }
