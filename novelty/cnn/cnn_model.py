import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from novelty.cnn.aggregator import *


class Novelty_CNN_conf:
    num_sent = 100
    filter_sizes = [4, 6, 9]
    num_filters = 100
    activation = "relu"
    dropout = 0.3
    freeze_encoder = False
    expand_features = True
    encoder_dim = 800

    def __init__(self, num_sent, encoder, **kwargs):
        self.num_sent = num_sent
        self.encoder = encoder
        for k, v in kwargs.items():
            setattr(self, k, v)


class DeepNoveltyCNN(nn.Module):
    def __init__(self, conf):
        super(DeepNoveltyCNN, self).__init__()
        self.accumulator = Accumulator(conf)
        self.linear = nn.Linear(conf.num_filters * len(conf.filter_sizes), 2)
        self.convs1 = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    conf.num_filters,
                    (K, conf.encoder_dim * 2 *(4 if conf.expand_features else 2)),
                )
                for K in conf.filter_sizes
            ]
        )
        if conf.activation.lower() == "relu".lower():
            self.act = nn.ReLU()
        elif conf.activation.lower() == "tanh".lower():
            self.act = nn.Tanh()
        elif conf.activation.lower() == "leakyrelu".lower():
            self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(conf.dropout)

    def forward(self, x, y):
        rdv = self.accumulator(x, y)
        opt = [self.act(conv(rdv.unsqueeze(1))).squeeze(3) for conv in self.convs1]
        opt = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in opt]
        opt = torch.cat(opt, 1)
        opt = self.act(opt)
        opt = self.linear(opt)
        return opt
