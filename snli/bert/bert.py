import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel,DistilBertModel


class Bert_Encoder_conf:
    encoder_dim = 768
    encoder = None
    batch_size = None
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Bert_Encoder(nn.Module):
    def __init__(self,conf):
        super(Bert_Encoder,self).__init__()
        self.conf=conf
        self.bert = conf.encoder
        self.fc = nn.Linear(conf.encoder_dim,3)

    def forward(self,x0):
        enc = self.bert.forward(x0)[0][:, 0, :]
        opt = self.fc(enc)
        opt = opt.unsqueeze(0)
        return opt