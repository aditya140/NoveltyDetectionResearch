import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
Implementation of GAN for supervised learning. Following the idea of SS-GAN or GAN-BERT,
we use Hierarchial attention networks to classify document pairs as novel or non-novel.

"""


class HAN_GAN_encoder_conf:
    encoder_dim = 600
    dropout = 0.3
    fc_hidden = 600
    freeze_encoder = False

    def __init__(self, encoder, **kwargs):
        self.encoder = encoder
        for k, v in kwargs.items():
            setattr(self, k, v)


class HAN_GAN_encoder(nn.Module):
    def __init__(self, conf):
        super(HAN_GAN_encoder, self).__init__()
        self.doc_encoder = conf.encoder
        self.doc_encoder.requires_grad = conf.freeze_encoder
        self.fc_in = nn.Linear(conf.encoder_dim * 4, conf.fc_hidden)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf.dropout)

    def forward(self, x0, x1):
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
        return cont


class Generator_conf:
    opt_dim = 600 * 4
    latent_dim = 100
    hidden_size = 600
    num_hidden_layers = 2
    dropout = 0.1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()
        self.conf = conf
        self.fc_in = nn.Linear(conf.latent_dim, conf.hidden_size)
        self.fcs = nn.ModuleList(
            [nn.Linear(conf.hidden_size, conf.hidden_size) for i in range(conf.num_hidden_layers)]
        )
        self.fc_out = nn.Linear(conf.hidden_size, conf.opt_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf.dropout)

    def forward(self, z):
        opt = self.dropout(self.act(self.fc_in(z)))
        for fc in self.fcs:
            opt = self.dropout(self.act(fc(opt)))
        opt = self.fc_out(opt)
        return opt


class Discriminator_conf:
    inp_dim = 600 * 4
    opt_classes = 2
    hidden_size = 600
    num_hidden_layers = 2
    dropout = 0.1
    encoder = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Discriminator(nn.Module):
    def __init__(self, conf):
        super(Discriminator, self).__init__()
        self.conf = conf
        self.encoder = conf.encoder
        self.fc_in = nn.Linear(conf.inp_dim, conf.hidden_size)
        self.fcs = nn.ModuleList(
            [nn.Linear(conf.hidden_size, conf.hidden_size) for i in range(conf.num_hidden_layers)]
        )
        self.fc_out = nn.Linear(conf.hidden_size, conf.opt_classes + 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf.dropout)
        self.softmax = nn.Softmax()

    def forward(self, z):
        features = self.dropout(self.act(self.fc_in(z)))
        for fc in self.fcs:
            features = self.dropout(self.act(fc(features)))
        logits = self.fc_out(features)
        prob = self.softmax(logits)
        return features, logits, prob
