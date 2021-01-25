import os
from snli.bilstm.bilstm import Bi_LSTM_Encoder_conf, Bi_LSTM_Encoder
from snli.attn_enc.attn_enc import Attn_Encoder_conf, Attn_Encoder
from novelty.han.han_novelty import HAN_Novelty, HAN_Novelty_conf
from document.han.han import HAN_conf, HAN
from lang import BertLangConf, GloveLangConf, LanguageIndex
from transformers import BertModel, DistilBertModel
import random
import torch
import joblib
import pickle

"""
Load Encoders
"""


def reset_model(model):
    """Parameters of each layer in the model will be reset. Only the layers which have an property reset_paramters will be reset.

    Args:
        model (nn.Module): Model parameters to be reset

    Returns:
        nn.Module: Model with parameters reset
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    return model


def load_bert_encoder():
    """Load Bert encoder from the models path

    Returns:
        [nn.Module]: Bert encoder
        [LanguageIndex]: Language Index
    """
    BERT_ENC_PATH = f"./models/bert_encoder/"
    with open(BERT_ENC_PATH + "model_conf.pkl", "rb") as f:
        model_conf = pickle.load(f)
    with open(BERT_ENC_PATH + "lang.pkl", "rb") as f:
        Lang = joblib.load(f)
    encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
    encoder.load_state_dict(torch.load(BERT_ENC_PATH + "weights.pt"))
    return encoder, Lang


def load_han_bilstm_encoder(id):
    """Load Hierarchical Attention Network encoder(trained on regression task) from the models path

    Returns:
        [nn.Module]: HAN encoder
        [LanguageIndex]: Language Index
    """
    HAN_PATH = f"./models/han/{id}/"
    with open(HAN_PATH + "model_conf.pkl", "rb") as f:
        model_conf = pickle.load(f)
    attn_enc, Lang = load_bilstm_encoder("SNLI-14")
    model_conf.encoder = attn_enc
    encoder = HAN(model_conf)
    encoder.load_state_dict(torch.load(HAN_PATH + "weights.pt"))
    return encoder, Lang


def load_han_attn_encoder(id):
    """Load Hierarchical Attention Network encoder(trained on classification task) from the models path

    Returns:
        [nn.Module]: HAN encoder
        [LanguageIndex]: Language Index
    """
    HAN_PATH = f"./models/han/{id}/"
    with open(HAN_PATH + "model_conf.pkl", "rb") as f:
        model_conf = pickle.load(f)
    attn_enc, Lang = load_attn_encoder("SNLI-12")
    model_conf.encoder = attn_enc
    encoder = HAN(model_conf)
    encoder.load_state_dict(torch.load(HAN_PATH + "weights.pt"))
    return encoder, Lang


def load_bilstm_encoder(id):
    """Load BiLSTM encoder from the models path

    Returns:
        [nn.Module]: BiLSTM encoder
        [LanguageIndex]: Language Index
    """
    BILSTM_PATH = f"./models/bilstm_encoder/{id}/"
    with open(BILSTM_PATH + "model_conf.pkl", "rb") as f:
        model_conf = pickle.load(f)
    with open(BILSTM_PATH + "lang.pkl", "rb") as f:
        Lang = joblib.load(f)
    encoder = Bi_LSTM_Encoder(model_conf)
    encoder.load_state_dict(torch.load(BILSTM_PATH + "weights.pt"))
    return encoder, Lang


def load_attn_encoder(id):
    """Load Attention encoder from the models path

    Returns:
        [nn.Module]: Attention LSTM encoder
        [LanguageIndex]: Language Index
    """
    ATTN_ENC_PATH = f"./models/attn_encoder/{id}/"
    with open(ATTN_ENC_PATH + "model_conf.pkl", "rb") as f:
        model_conf = pickle.load(f)
    with open(ATTN_ENC_PATH + "lang.pkl", "rb") as f:
        Lang = joblib.load(f)
    encoder = Attn_Encoder(model_conf)
    encoder.load_state_dict(torch.load(ATTN_ENC_PATH + "weights.pt"))
    return encoder, Lang


def load_han_novelty(id):
    """Load Hierarchical Attention Network encoder(trained on classification task) from the models path

    Returns:
        [nn.Module]: HAN encoder
        [LanguageIndex]: Language Index
    """
    HAN_NOV_PATH = f"./models/han_novelty/{id}/"
    with open(HAN_NOV_PATH + "model_conf.pkl", "rb") as f:
        model_conf = pickle.load(f)
    attn_enc, Lang = load_han_attn_encoder("DOC-13")
    model_conf.encoder = attn_enc
    encoder = HAN_Novelty(model_conf)
    encoder.load_state_dict(torch.load(HAN_NOV_PATH + "weights.pt"))
    return encoder, Lang, model_conf
