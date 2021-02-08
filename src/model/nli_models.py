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
        opt = torch.tanh(opt)
        opt = self.Wa(opt)
        opt = F.softmax(opt, dim=1)
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
            batch_first=True,
        )
        self.attention = Attention(conf)

    def forward(self, inp):
        batch_size = inp.shape[0]
        embedded = self.embedding(inp)
        embedded = self.relu(self.translate(embedded))
        all_, (_, _) = self.lstm_layer(embedded)
        attn = self.attention(all_)
        cont = torch.bmm(attn.permute(0, 2, 1), all_)
        cont = cont.squeeze(1)
        return cont


class AttnBiLSTM_snli(nn.Module):
    def __init__(self, conf):
        super(AttnBiLSTM_snli, self).__init__()
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
        x0_enc = self.encoder(x0)
        x1_enc = self.encoder(x1)
        cont = torch.cat(
            [x0_enc, x1_enc, torch.abs(x0_enc - x1_enc), x0_enc * x1_enc], dim=1
        )
        opt = self.fc_in(cont)
        opt = self.dropout(opt)
        for fc in self.fcs:
            opt = self.relu(self.dropout(fc(opt)))
        opt = self.fc_out(opt)
        return opt


def attn_bilstm_snli(options):
    return AttnBiLSTM_snli(options)


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
            batch_first=True,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=conf["dropout"])

    def forward(self, x):
        batch_size = x.shape[0]
        embed = self.embedding(x)
        proj = self.relu(self.projection(embed))
        _, (hid, _) = self.lstm(proj)
        hid = hid[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
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


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Structured Self attention based SNLI model
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Struc_Attention(nn.Module):
    def __init__(self, conf):
        super(Struc_Attention, self).__init__()
        self.Ws = nn.Linear(
            2 * conf["hidden_size"],
            conf["attention_layer_param"],
            bias=False,
        )
        self.Wa = nn.Linear(conf["attention_layer_param"], conf["r"], bias=False)

    def forward(self, hid):
        opt = self.Ws(hid)
        opt = torch.tanh(opt)
        opt = self.Wa(opt)
        opt = F.softmax(opt, dim=1)
        return opt


class Struc_Attn_Encoder(nn.Module):
    def __init__(self, conf):
        super(Struc_Attn_Encoder, self).__init__()
        self.conf = conf
        self.embedding = nn.Embedding(
            num_embeddings=conf["vocab_size"],
            embedding_dim=conf["embedding_dim"],
            padding_idx=conf["padding_idx"],
        )
        if conf["use_glove"]:
            self.embedding = nn.Embedding.from_pretrained(
                torch.load(".vector_cache/{}_vectors.pt".format(conf["dataset"]))
            )
        self.translate = nn.Linear(self.conf["embedding_dim"], self.conf["hidden_size"])
        self.relu = nn.ReLU()

        self.lstm_layer = nn.LSTM(
            input_size=self.conf["hidden_size"],
            hidden_size=self.conf["hidden_size"],
            num_layers=self.conf["num_layers"],
            bidirectional=True,
            batch_first=True,
        )
        self.attention = Struc_Attention(conf)

    def forward(self, inp):
        batch_size = inp.shape[0]
        embedded = self.embedding(inp)
        embedded = self.relu(self.translate(embedded))
        all_, (_, _) = self.lstm_layer(embedded)
        attn = self.attention(all_)
        cont = torch.bmm(attn.permute(0, 2, 1), all_)
        return cont, attn


class Struc_Attn_encoder_snli(nn.Module):
    def __init__(self, conf):
        super(Struc_Attn_encoder_snli, self).__init__()
        self.conf = conf
        self.encoder = Struc_Attn_Encoder(conf)
        self.gated = self.conf["gated"]
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.pool_strategy = self.conf["pool_strategy"]
        # Gated parameters
        if self.gated:
            self.wt_p = torch.nn.Parameter(
                torch.rand(
                    (
                        self.conf["r"],
                        2 * self.conf["hidden_size"],
                        self.conf["gated_embedding_dim"],
                    )
                )
            )
            self.wt_h = torch.nn.Parameter(
                torch.rand(
                    (
                        self.conf["r"],
                        2 * self.conf["hidden_size"],
                        self.conf["gated_embedding_dim"],
                    )
                )
            )
            self.init_gated_encoder()
            self.fc_in = nn.Linear(
                self.conf["gated_embedding_dim"] * self.conf["r"],
                self.conf["hidden_size"],
            )
            self.fcs = nn.ModuleList(
                [
                    nn.Linear(self.conf["hidden_size"], self.conf["hidden_size"])
                    for i in range(self.conf["fcs"])
                ]
            )
            self.fc_out = nn.Linear(self.conf["hidden_size"], 3)

        # Non Gated Version (max pool avg pool)
        else:
            self.fc_in = nn.Linear(
                2 * 4 * self.conf["hidden_size"],
                self.conf["hidden_size"],
            )
            self.fcs = nn.ModuleList(
                [
                    nn.Linear(
                        self.conf["hidden_size"],
                        self.conf["hidden_size"],
                    )
                    for i in range(self.conf["fcs"])
                ]
            )
            self.fc_out = nn.Linear(self.conf["hidden_size"], 3)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=self.conf["dropout"])

    def init_gated_encoder(self):
        nn.init.kaiming_uniform_(self.wt_p)
        nn.init.kaiming_uniform_(self.wt_h)

    def penalty_l2(self, att):
        att = att.permute(1, 0, 2)
        penalty = (
            torch.norm(
                torch.bmm(att, att.transpose(1, 2))
                - torch.eye(att.size(1)).to(self.template.device),
                p="fro",
            )
            / att.size(0)
        ) ** 2
        return penalty

    def forward(self, x0, x1):
        x0_enc, x0_attn = self.encoder(x0)
        x0_enc = self.dropout(x0_enc)
        x1_enc, x1_attn = self.encoder(x1)
        x1_enc = self.dropout(x1_enc)

        if self.gated:
            F0 = x0_enc @ self.wt_p
            F1 = x1_enc @ self.wt_h
            Fr = F0 * F1
            Fr = Fr.permute(1, 0, 2).flatten(start_dim=1)
        else:
            if self.pool_strategy == "avg":
                F0 = x0_enc.mean(1)
                F1 = x1_enc.mean(1)
                Fr = torch.cat([F0, F1, torch.abs(F0 - F1), F0 * F1], dim=1)
            elif self.pool_strategy == "max":
                F0 = x0_enc.max(1)
                F0 = F0.values
                F1 = x1_enc.max(1)
                F1 = F1.values
                Fr = torch.cat([F0, F1, torch.abs(F0 - F1), F0 * F1], dim=1)

        opt = self.fc_in(Fr)
        opt = self.dropout(opt)
        for fc in self.fcs:
            opt = fc(opt)
            opt = self.dropout(opt)
            opt = self.act(opt)
        opt = self.fc_out(opt)
        return opt


def struc_attn_snli(options):
    return Struc_Attn_encoder_snli(options)


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
BERT based SNLI model
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

from transformers import BertModel, DistilBertModel


class Bert_Encoder(nn.Module):
    def __init__(self, conf):
        super(Bert_Encoder, self).__init__()
        self.conf = conf
        self.bert = conf.encoder
        self.fc = nn.Linear(conf.encoder_dim, 3)

    def forward(self, x0):
        enc = self.bert.forward(x0)[0][:, 0, :]
        opt = self.fc(enc)
        opt = opt.unsqueeze(0)
        return opt