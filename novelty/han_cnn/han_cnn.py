import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from novelty.cnn.aggregator import *


class HAN_CNN_conf:
    num_sent = 100
    filter_sizes = [4, 6, 9]
    num_filters = 100
    activation = "relu"
    dropout = 0.3
    freeze_encoder = False
    expand_features = True
    encoder_dim = 400
    doc_encoder_dim = 600
    fc_hidden = 600
    freeze_encoder = False

    def __init__(self, num_sent, encoder, **kwargs):
        self.num_sent = num_sent
        self.encoder = encoder
        for k, v in kwargs.items():
            setattr(self, k, v)


class HAN_CNN(nn.Module):
    def __init__(self, conf):
        super(HAN_CNN, self).__init__()
        self.accumulator = Accumulator(conf)
        self.convs1 = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    conf.num_filters,
                    (K, conf.encoder_dim * 2 * (4 if conf.expand_features else 2)),
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

        self.doc_encoder = conf.encoder
        self.doc_encoder.requires_grad = conf.freeze_encoder
        self.fc_in = nn.Linear(conf.doc_encoder_dim * 4, conf.fc_hidden)
        self.dropout = nn.Dropout(conf.dropout)
        self.fc_final = nn.Linear(conf.num_filters * len(conf.filter_sizes) + conf.fc_hidden, 2)

    def forward(self, x0, x1):
        rdv = self.accumulator(x0, x1)
        opt = [self.act(conv(rdv.unsqueeze(1))).squeeze(3) for conv in self.convs1]
        opt = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in opt]
        opt = torch.cat(opt, 1)
        opt = self.dropout(self.act(opt))

        x0_enc = self.doc_encoder(x0)
        x1_enc = self.doc_encoder(x1)
        cont = torch.cat(
                [
                    x0_enc,
                    x1_enc,
                    torch.abs(x0_enc - x1_enc),
                    x0_enc * x1_enc,
                ],
                dim=1,
            )
        cont = self.dropout(cont)
        cont = self.dropout(self.act(self.fc_in(cont)))
        final = torch.cat([cont,opt],dim=1)
        return self.fc_final(final)
