import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
BiLSTM + Attention based SNLI model
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class Attention(nn.Module):
    def __init__(self, conf):
        super(Attention, self).__init__()
        self.Ws = nn.Linear(
            2 * conf["hidden_size"],
            conf["attention_layer_param"],
            bias=False,
        )
        self.Wa = nn.Linear(conf["attention_layer_param"], 1, bias=False)

    def forward(self, hid):
        opt = self.Ws(hid)
        opt = F.tanh(opt)
        opt = self.Wa(opt)
        opt = F.softmax(opt)
        return opt


class Attn_Encoder(nn.Module):
    def __init__(self, conf):
        super(Attn_Encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=conf["vocab_size"],
            embedding_dim=conf["embedding_dim"],
            padding_idx=conf["padding_idx"],
        )
        self.translate = nn.Linear(conf["embedding_dim"], conf["hidden_size"])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=conf["dropout"])

        if conf["use_glove"]:
            self.embedding = nn.Embedding.from_pretrained(
                torch.load(".vector_cache/{}_vectors.pt".format(conf["dataset"]))
            )
        self.lstm_layer = nn.LSTM(
            input_size=conf["hidden_size"],
            hidden_size=conf["hidden_size"],
            num_layers=conf["num_layers"],
            dropout=conf["dropout"],
            bidirectional=True,
        )
        self.attention = Attention(conf)

    def forward(self, inp):
        batch_size = inp.shape[0]
        embedded = self.embedding(inp)
        embedded = self.translate(embedded)
        embedded = self.relu(embedded)
        embedded = embedded.permute(1, 0, 2)
        all_, (hid, cell) = self.lstm_layer(embedded)
        attn = self.attention(all_)
        cont = torch.bmm(all_.permute(1, 2, 0), attn.permute(1, 0, 2)).permute(2, 0, 1)
        return cont


class Attn_encoder_snli(nn.Module):
    def __init__(self, conf):
        super(Attn_encoder_snli, self).__init__()
        self.conf = conf
        self.encoder = Attn_Encoder(conf)
        self.fc_in = nn.Linear(
            2 * 4 * self.conf["hidden_size"],
            self.conf["hidden_size"],
        )
        self.fcs = nn.ModuleList(
            [
                nn.Linear(self.conf["hidden_size"], self.conf["hidden_size"])
                for i in range(self.conf["fcs"])
            ]
        )
        self.fc_out = nn.Linear(self.conf["hidden_size"], 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=self.conf["dropout"])

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
            opt = self.relu(opt)
        opt = self.fc_out(opt)
        return opt


def attn_bilstm_snli(options):
    return Attn_encoder_snli(options)


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
BiLSTM based SNLI model
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class BiLSTM_encoder(nn.Module):
    def __init__(self, conf):
        super(BiLSTM_encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=conf["vocab_size"],
            embedding_dim=conf["embedding_dim"],
            padding_idx=conf["padding_idx"],
        )
        if conf["use_glove"]:
            self.embedding = nn.Embedding.from_pretrained(
                torch.load(".vector_cache/{}_vectors.pt".format(conf["dataset"]))
            )
        self.projection = nn.Linear(conf["embedding_dim"], conf["hidden_size"])
        self.lstm = nn.LSTM(
            input_size=conf["hidden_size"],
            hidden_size=conf["hidden_size"],
            num_layers=conf["num_layers"],
            dropout=conf["dropout"],
            bidirectional=True,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=conf["dropout"])

    def forward(self, x):
        batch_size = x.shape[0]
        embed = self.embedding(x)
        proj = self.relu(self.projection(embed))
        proj = proj.permute(1, 0, 2)
        _, (hid, _) = self.lstm(proj)
        hid = hid.view(batch_size, -1)
        return hid


class BiLSTM_snli(nn.Module):
    def __init__(self, conf):
        super(BiLSTM_snli, self).__init__()
        self.encoder = BiLSTM_encoder(conf)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=conf["dropout"])

        self.lin1 = nn.Linear(conf["hidden_size"] * 2 * 4, conf["hidden_size"])
        self.lin2 = nn.Linear(conf["hidden_size"], conf["hidden_size"])
        self.lin3 = nn.Linear(conf["hidden_size"], 3)

        for lin in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.out = nn.Sequential(
            self.lin1,
            self.relu,
            self.dropout,
            self.lin2,
            self.relu,
            self.dropout,
            self.lin3,
        )

    def forward(self, x0, x1):
        x0 = self.encoder(x0)
        x1 = self.encoder(x1)

        combined = torch.cat(
            (
                x0,
                x1,
                torch.abs(x0 - x1),
                x0 * x1,
            ),
            dim=1,
        )
        return self.out(combined)


def bilstm_snli(options):
    return BiLSTM_snli(options)
