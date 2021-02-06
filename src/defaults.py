from argparse import ArgumentParser
import os
import random
import numpy as np
import torch


# snli_spacy_conf = {
#     "batch_size": 128,
#     "max_len": 40,
#     "device": "cuda",
#     "tokenizer": "spacy",
# }
# snli_bert_conf = {
#     "batch_size": 128,
#     "max_len": 40,
#     "device": "cuda",
#     "tokenizer": "bert",
# }

# model_conf = {
#     "hidden_size": 300,
#     "embedding_dim": 300,
#     "dropout": 0.3,
#     "use_glove": True,
#     "num_layers": 1,
#     "dataset": "snli",
#     "fcs": 1,
#     "vocab_size": dataset.vocab_size(),
#     "tokenizer": "spacy",
#     "padding_idx": dataset.padding_idx(),
#     "attention_layer_param": 200,
# }


# hparams = {
#     "optimizer_base": {
#         "optim": "adamw",
#         "lr": 0.0010039910781394373,
#         "scheduler": "const",
#     },
#     "optimizer_tune": {
#         "optim": "adam",
#         "lr": 0.0010039910781394373,
#         "weight_decay": 0.1,
#         "scheduler": "lambda",
#     },
#     "switch_epoch": 5,
# }


NEPTUNE_API = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTg3MzU5NjQtMmIxZC00Njg0LTgzYzMtN2UwYjVlYzVhNDg5In0="


def parse_nli_conf():
    parser = ArgumentParser(description="PyTorch/torchtext NLI Baseline")
    parser.add_argument("--dataset", "-d", type=str, default="mnli")

    # language
    parser.add_argument("--tokenizer", type=str, default="bert")
    parser.add_argument("--max_len", type=int, default=50)
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

    parser.add_argument("--results_dir", type=str, default="results")
    return check_args(parser.parse_args())


def attention_model_params(parser_dump):
    parser_dump.add_argument("--hidden_size", type=int, default=400)
    parser_dump.add_argument("--embedding_dim", type=int, default=300)
    parser_dump.add_argument("--dropout", type=float, default=0.3)
    parser_dump.add_argument("--use_glove", type=bool, default=False)
    parser_dump.add_argument("--num_layers", type=int, default=1)
    parser_dump.add_argument("--fcs", type=int, default=1)
    parser_dump.add_argument("--attention_layer_param", type=int, default=200)


def bilstm_model_params(parser_dump):
    parser_dump.add_argument("--hidden_size", type=int, default=400)
    parser_dump.add_argument("--embedding_dim", type=int, default=300)
    parser_dump.add_argument("--dropout", type=float, default=0.3)
    parser_dump.add_argument("--use_glove", type=bool, default=True)
    parser_dump.add_argument("--num_layers", type=int, default=1)
    parser_dump.add_argument("--fcs", type=int, default=1)
    parser_dump.add_argument("--attention_layer_param", type=int, default=200)


def check_args(args):
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


def get_conf(args):
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
