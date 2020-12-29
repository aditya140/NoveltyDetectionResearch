from novelty.cnn.aggregator import *
from novelty.cnn.cnn_model import *
from snli.bilstm.bilstm import *
import pickle
import joblib
from utils import load_bilstm_encoder, load_attn_encoder
import torch
import shutil
import os
from tqdm.auto import tqdm
import math




def batch(iterable, n = 32):
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch

def load_novelty_cnn(encoder_type="attention"):
    MODEL_PATH = f"./models/cnn_novelty/"
    with open(MODEL_PATH + "model_conf.pkl", "rb") as f:
        model_conf = pickle.load(f)
    with open(MODEL_PATH + "lang.pkl", "rb") as f:
        Lang = joblib.load(f)
    if encoder_type == "bilstm":
        enc, _ = load_bilstm_encoder()
    if encoder_type == "attention":
        enc, _ = load_attn_encoder()
    model_conf.encoder = enc
    model = DeepNoveltyCNN(model_conf)
    model.load_state_dict(torch.load(MODEL_PATH + "weights.pt"))
    model.eval()
    return {"model": model, "lang": Lang, "model_conf": model_conf}


def to_novelty_tensor(s, Lang, model_conf):
    if isinstance(s, list):
        s = ". ".join(s)
    pad_arr = [0] * Lang.max_len
    t = Lang.encode_batch(Lang.preprocess_sentence(s).split("."))
    if t.shape[0] < model_conf.num_sent:
        opt = np.append(t, [pad_arr] * (model_conf.num_sent - t.shape[0]), axis=0)
    else:
        opt = t[
            : model_conf.num_sent,
        ]
    return opt


def predict_novelty(source, target, model, cuda=True, max_batch_size = 32):
    opt = []
    for inp_batch in tqdm(batch(list(zip(source,target)),n=32),total=math.ceil(len(source)/32)):
        src,trg = map(list,zip(*inp_batch))
        src_vec = []
        for i in src:
            src_vec.append(torch.tensor(
                to_novelty_tensor(i, model["lang"], model["model_conf"])
            ))
        trg_vec = []
        for j in trg:
            trg_vec.append(torch.tensor(
                to_novelty_tensor(j, model["lang"], model["model_conf"])
            ))
        src_vec = torch.stack(src_vec)
        trg_vec = torch.stack(trg_vec)

        if cuda:
            src_vec = src_vec.cuda()
            trg_vec = trg_vec.cuda()
        opt+=(
            torch.nn.functional.softmax(model["model"](trg_vec, src_vec))
            .cpu()
            .detach()
            .tolist()
        )
    return opt

