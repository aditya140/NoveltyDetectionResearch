import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sru import SRU, SRUCell


class SRU_Encoder_conf:
    embedding_dim = 300
    hidden_size = 300
    fcs = 1
    num_layers = 1
    dropout = 0.1
    opt_labels = 3
    bidirectional = True
    activation = "tanh"
    freeze_embedding = False

    def __init__(self, lang, embedding_matrix=None, **kwargs):
        self.embedding_matrix = embedding_matrix
        self.vocab_size = lang.vocab_size_final()
        self.padding_idx = 0
        for k, v in kwargs.items():
            setattr(self, k, v)


class SRU_Encoder(nn.Module):
    def __init__(self, conf):
        super(SRU_Encoder, self).__init__()
        self.conf = conf
        self.embedding = nn.Embedding(
            num_embeddings=self.conf.vocab_size,
            embedding_dim=self.conf.embedding_dim,
            padding_idx=self.conf.padding_idx,
        )
        self.translate = nn.Linear(300, self.conf.hidden_size)
        if self.conf.activation.lower() == "relu".lower():
            self.act = nn.ReLU()
        elif self.conf.activation.lower() == "tanh".lower():
            self.act = nn.Tanh()
        elif self.conf.activation.lower() == "leakyrelu".lower():
            self.act = nn.LeakyReLU()
        if isinstance(self.conf.embedding_matrix, np.ndarray):
            self.embedding.from_pretrained(
                torch.tensor(self.conf.embedding_matrix),
                freeze=self.conf.freeze_embedding,
            )
        self.sru_layer = nn.LSTM(
            input_size=self.conf.hidden_size,
            hidden_size=self.conf.hidden_size,
            num_layers=self.conf.num_layers,
            bidirectional=self.conf.bidirectional,
        )

    def forward(self, inp):
        batch_size = inp.shape[0]
        embedded = self.embedding(inp)
        embedded = self.translate(embedded)
        embedded = self.act(embedded)
        embedded = embedded.permute(1, 0, 2)
        _, (hid, cell) = self.sru_layer(embedded)
        if self.conf.bidirectional:
            hid = torch.cat((hid[-1], hid[-2]), 1)
        else:
            hid = hid[-1]
        hid = hid.unsqueeze(0)
        cont = hid.permute(0, 1, 2)
        return cont


class SRU_SNLI(nn.Module):
    def __init__(self, conf):
        super(SRU_SNLI, self).__init__()
        self.conf = conf
        self.encoder = SRU_Encoder(conf)
        self.fc_in = nn.Linear(
            (2 if conf.bidirectional else 1) * 4 * self.conf.hidden_size,
            self.conf.hidden_size,
        )
        self.fcs = nn.ModuleList(
            [
                nn.Linear(self.conf.hidden_size, self.conf.hidden_size)
                for i in range(self.conf.fcs)
            ]
        )
        self.fc_out = nn.Linear(self.conf.hidden_size, self.conf.opt_labels)
        if self.conf.activation.lower() == "relu".lower():
            self.act = nn.ReLU()
        elif self.conf.activation.lower() == "tanh".lower():
            self.act = nn.Tanh()
        elif self.conf.activation.lower() == "leakyrelu".lower():
            self.act = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=self.conf.dropout)

    def forward(self, x0, x1):
        x0_enc = self.encoder(x0.long())
        x0_enc = self.dropout(x0_enc)
        x1_enc = self.encoder(x1.long())
        x1_enc = self.dropout(x1_enc)
        cont = torch.cat(
            [x0_enc, x1_enc, torch.abs(x0_enc - x1_enc), x0_enc * x1_enc], dim=2
        )
        opt = self.fc_in(cont)
        opt = self.dropout(opt)
        for fc in self.fcs:
            opt = fc(opt)
            opt = self.dropout(opt)
            opt = self.act(opt)
        opt = self.fc_out(opt)
        return opt
