from argparse import ArgumentParser
import os
import random
import numpy as np
import torch
import logging
import time
import dill
import neptune
import shutil
import dill


NEPTUNE_API = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTg3MzU5NjQtMmIxZC00Njg0LTgzYzMtN2UwYjVlYzVhNDg5In0="
NLI_NEPTUNE_PROJECT = "aparkhi/NLI"
NOVELTY_NEPTUNE_PROJECT = "aparkhi/Novelty"


"""
Argument Parser
"""


def parse_nli_conf():
    parser = ArgumentParser(description="PyTorch/torchtext NLI Baseline")
    parser.add_argument("--dataset", "-d", type=str, default="mnli")

    # language
    parser.add_argument("--tokenizer", type=str, default="bert")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_char_emb", type=bool, default=False)
    parser.add_argument("--max_word_len", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    # optimizer_conf
    parser.add_argument("--epochs", type=int, default=20)
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
    return args


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


def parse_novelty_conf():
    parser = ArgumentParser(description="PyTorch/torchtext Novelty Training")
    parser.add_argument("--dataset", "-d", type=str, default="dlnd")

    # language
    parser.add_argument("--load_nli", type=str, default="None")
    parser.add_argument("--max_num_sent", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tokenizer", type=str, default="None")
    parser.add_argument("--max_len", type=int, default=0)
    parser.add_argument("--sent_tokenizer", type=str, default="spacy")

    # optimizer_conf
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--loss_agg", type=str, default="sum")
    parser.add_argument("--scheduler", type=str, default="step")

    subparsers = parser.add_subparsers(dest="model_type")

    # model_conf
    parser_dan = subparsers.add_parser("dan")
    dan_model_parameters(parser_dan)

    # model_conf
    parser_han = subparsers.add_parser("han")
    han_model_parameters(parser_han)

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


def dan_model_parameters(parser_dump):
    parser_dump.add_argument("--hidden_size", type=int, default=400)
    parser_dump.add_argument("--dropout", type=float, default=0.3)
    parser_dump.add_argument("--use_glove", type=bool, default=False)


def han_model_parameters(parser_dump):
    pass


"""
Utils
"""


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
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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