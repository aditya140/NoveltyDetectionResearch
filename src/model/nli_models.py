import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from transformers import BertModel, DistilBertModel


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
        self.translate = nn.Linear(
            (
                conf["embedding_dim"]
                + int(conf["use_char_emb"]) * conf["char_embedding_dim"]
            ),
            conf["hidden_size"],
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=conf["dropout"])

        if conf["use_glove"]:
            self.embedding = nn.Embedding.from_pretrained(
                torch.load(".vector_cache/{}_vectors.pt".format(conf["dataset"]))
            )

        if conf["use_char_emb"]:
            self.char_embedding = nn.Embedding(
                num_embeddings=conf["char_vocab_size"],
                embedding_dim=conf["char_embedding_dim"],
                padding_idx=0,
            )
            self.char_cnn = nn.Conv2d(
                conf["max_word_len"],
                conf["char_embedding_dim"],
                (1, 6),
                stride=(1, 1),
                padding=0,
                bias=True,
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

    def char_embedding_forward(self, x):
        # X - [batch_size, seq_len, char_emb_size])
        batch_size, seq_len, char_emb_size = x.shape
        x = x.view(-1, char_emb_size)
        x = self.char_embedding(x)  # (batch_size * seq_len, char_emb_size, emb_size)
        x = x.view(batch_size, -1, seq_len, char_emb_size)
        x = x.permute(0, 3, 2, 1)
        x = self.char_cnn(x)
        x = torch.max(F.relu(x), 3)[0]
        return x.view(batch_size, seq_len, -1)

    def forward(self, inp, char_vec):
        batch_size = inp.shape[0]
        embedded = self.embedding(inp)
        if char_vec != None:
            char_emb = self.char_embedding_forward(char_vec)
            embedded = torch.cat([embedded, char_emb], dim=2)
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

    def forward(self, x0, x1, **kwargs):
        char_vec_x0 = kwargs.get("char_premise", None)
        char_vec_x1 = kwargs.get("char_hypothesis", None)
        x0_enc = self.encoder(x0, char_vec_x0)
        x1_enc = self.encoder(x1, char_vec_x1)
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

        if conf["use_char_emb"]:
            self.char_embedding = nn.Embedding(
                num_embeddings=conf["char_vocab_size"],
                embedding_dim=conf["char_embedding_dim"],
                padding_idx=0,
            )
            self.char_cnn = nn.Conv2d(
                conf["max_word_len"],
                conf["char_embedding_dim"],
                (1, 6),
                stride=(1, 1),
                padding=0,
                bias=True,
            )

        self.projection = nn.Linear(
            (
                conf["embedding_dim"]
                + int(conf["use_char_emb"]) * conf["char_embedding_dim"]
            ),
            conf["hidden_size"],
        )
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

    def char_embedding_forward(self, x):
        # X - [batch_size, seq_len, char_emb_size])
        batch_size, seq_len, char_emb_size = x.shape
        x = x.view(-1, char_emb_size)
        x = self.char_embedding(x)  # (batch_size * seq_len, char_emb_size, emb_size)
        x = x.view(batch_size, -1, seq_len, char_emb_size)
        x = x.permute(0, 3, 2, 1)
        x = self.char_cnn(x)
        x = torch.max(F.relu(x), 3)[0]
        return x.view(batch_size, seq_len, -1)

    def forward(self, x, char_vec):
        batch_size = x.shape[0]
        embedded = self.embedding(x)
        if char_vec != None:
            char_emb = self.char_embedding_forward(char_vec)
            embedded = torch.cat([embedded, char_emb], dim=2)
        proj = self.relu(self.projection(embedded))
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

    def forward(self, x0, x1, **kwargs):

        char_vec_x0 = kwargs.get("char_premise", None)
        char_vec_x1 = kwargs.get("char_hypothesis", None)
        x0_enc = self.encoder(x0, char_vec_x0)
        x1_enc = self.encoder(x1, char_vec_x1)

        combined = torch.cat(
            (
                x0_enc,
                x1_enc,
                torch.abs(x0_enc - x1_enc),
                x0_enc * x1_enc,
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

        if conf["use_char_emb"]:
            self.char_embedding = nn.Embedding(
                num_embeddings=conf["char_vocab_size"],
                embedding_dim=conf["char_embedding_dim"],
                padding_idx=0,
            )
            self.char_cnn = nn.Conv2d(
                conf["max_word_len"],
                conf["char_embedding_dim"],
                (1, 6),
                stride=(1, 1),
                padding=0,
                bias=True,
            )
        self.translate = nn.Linear(
            (
                conf["embedding_dim"]
                + int(conf["use_char_emb"]) * conf["char_embedding_dim"]
            ),
            conf["hidden_size"],
        )

        self.relu = nn.ReLU()

        self.lstm_layer = nn.LSTM(
            input_size=self.conf["hidden_size"],
            hidden_size=self.conf["hidden_size"],
            num_layers=self.conf["num_layers"],
            bidirectional=True,
            batch_first=True,
        )
        self.attention = Struc_Attention(conf)

    def char_embedding_forward(self, x):
        # X - [batch_size, seq_len, char_emb_size])
        batch_size, seq_len, char_emb_size = x.shape
        x = x.view(-1, char_emb_size)
        x = self.char_embedding(x)  # (batch_size * seq_len, char_emb_size, emb_size)
        x = x.view(batch_size, -1, seq_len, char_emb_size)
        x = x.permute(0, 3, 2, 1)
        x = self.char_cnn(x)
        x = torch.max(F.relu(x), 3)[0]
        return x.view(batch_size, seq_len, -1)

    def forward(self, inp, char_vec):
        batch_size = inp.shape[0]
        embedded = self.embedding(inp)
        if char_vec != None:
            char_emb = self.char_embedding_forward(char_vec)
            embedded = torch.cat([embedded, char_emb], dim=2)
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

    def forward(self, x0, x1, **kwargs):

        char_vec_x0 = kwargs.get("char_premise", None)
        char_vec_x1 = kwargs.get("char_hypothesis", None)

        x0_enc, x0_attn = self.encoder(x0, char_vec_x0)
        x1_enc, x1_attn = self.encoder(x1, char_vec_x1)

        x0_enc = self.dropout(x0_enc)
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

from transformers import DistilBertModel


class Bert_Encoder(nn.Module):
    def __init__(self, conf):
        super(Bert_Encoder, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, x0):
        (enc,) = self.bert.forward(x0)
        enc = torch.mean(enc, dim=1)
        # enc = torch.max(enc,dim=1).values
        return enc


class Bert_snli(nn.Module):
    def __init__(self, conf):
        super(Bert_snli, self).__init__()
        self.bert = Bert_Encoder(conf)
        self.fc = nn.Linear(4 * 768, 3)

    def forward(self, x0, x1):
        x0_enc = self.bert(x0)
        x1_enc = self.bert(x1)
        enc = torch.cat(
            [x0_enc, x1_enc, torch.abs(x0_enc - x1_enc), x0_enc * x1_enc], dim=1
        )
        opt = self.fc(enc)
        return opt


def bert_snli(conf):
    return Bert_snli(conf)


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Multiway Attention Network
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class concat_attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wc1 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.Wc2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.vc = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, y):
        _s1 = self.Wc1(x).unsqueeze(1)
        _s2 = self.Wc2(y).unsqueeze(2)
        sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze()
        ait = F.softmax(sjt, 2)
        qtc = ait.bmm(x)
        return qtc


class bilinear_attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wb = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, x, y):
        _s1 = self.Wb(x).transpose(2, 1)
        sjt = y.bmm(_s1)
        ait = F.softmax(sjt, 2)
        qtb = ait.bmm(x)
        return qtb


class dot_attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wd = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.vd = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, y):
        _s1 = x.unsqueeze(1)
        _s2 = y.unsqueeze(2)
        sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qtd = ait.bmm(x)
        return qtd


class minus_attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wm = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.vm = nn.Linear(hidden_size, 1, bias=False)

        self.Ws = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.vs = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, y):
        _s1 = x.unsqueeze(1)
        _s2 = y.unsqueeze(2)
        sjt = self.vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qtm = ait.bmm(x)
        return qtm


class MwAN_snli(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.dropout = nn.Dropout(conf["dropout"])

        self.embedding = nn.Embedding(
            num_embeddings=conf["vocab_size"],
            embedding_dim=conf["embedding_dim"],
            padding_idx=conf["padding_idx"],
        )

        if conf["use_glove"]:
            self.embedding = nn.Embedding.from_pretrained(
                torch.load(".vector_cache/{}_vectors.pt".format(conf["dataset"]))
            )
            if conf["freeze_emb"]:
                self.embedding.requires_grad_(False)

        if conf["use_char_emb"]:
            self.char_embedding = nn.Embedding(
                num_embeddings=conf["char_vocab_size"],
                embedding_dim=conf["char_embedding_dim"],
                padding_idx=0,
            )
            self.char_cnn = nn.Conv2d(
                conf["max_word_len"],
                conf["char_embedding_dim"],
                (1, 6),
                stride=(1, 1),
                padding=0,
                bias=True,
            )

        self.prem_gru = nn.GRU(
            input_size=(
                conf["embedding_dim"]
                + int(conf["use_char_emb"]) * conf["char_embedding_dim"]
            ),
            hidden_size=conf["hidden_size"],
            batch_first=True,
            bidirectional=True,
        )

        self.hypo_gru = nn.GRU(
            input_size=(
                conf["embedding_dim"]
                + int(conf["use_char_emb"]) * conf["char_embedding_dim"]
            ),
            hidden_size=conf["hidden_size"],
            batch_first=True,
            bidirectional=True,
        )

        # Concat Attention
        self.concat_attn = concat_attention(conf["hidden_size"])
        # Bilinear Attention
        self.bilinear_attn = bilinear_attention(conf["hidden_size"])
        # Dot Attention :
        self.dot_attn_1 = dot_attention(conf["hidden_size"])
        self.dot_attn_2 = dot_attention(conf["hidden_size"])
        # Minus Attention :
        self.minus_attn = minus_attention(conf["hidden_size"])

        self.gru_agg = nn.GRU(
            12 * conf["hidden_size"],
            conf["hidden_size"],
            batch_first=True,
            bidirectional=True,
        )

        self.Wq = nn.Linear(2 * conf["hidden_size"], conf["hidden_size"], bias=False)
        self.vq = nn.Linear(conf["hidden_size"], 1, bias=False)

        self.Wp1 = nn.Linear(2 * conf["hidden_size"], conf["hidden_size"], bias=False)
        self.Wp2 = nn.Linear(2 * conf["hidden_size"], conf["hidden_size"], bias=False)
        self.vp = nn.Linear(conf["hidden_size"], 1, bias=False)

        self.prediction = nn.Linear(2 * conf["hidden_size"], 3, bias=False)

    def char_embedding_forward(self, x):
        # X - [batch_size, seq_len, char_emb_size])
        batch_size, seq_len, char_emb_size = x.shape
        x = x.view(-1, char_emb_size)
        x = self.char_embedding(x)  # (batch_size * seq_len, char_emb_size, emb_size)
        x = x.view(batch_size, -1, seq_len, char_emb_size)
        x = x.permute(0, 3, 2, 1)
        x = self.char_cnn(x)
        x = torch.max(F.relu(x), 3)[0]
        return x.view(batch_size, seq_len, -1)

    def forward(self, premise, hypothesis, **kwargs):
        char_vec_x0 = kwargs.get("char_premise", None)
        char_vec_x1 = kwargs.get("char_hypothesis", None)

        hp = self.embedding(premise)
        hh = self.embedding(hypothesis)

        if char_vec_x0 != None:
            char_emb_hp = self.char_embedding_forward(char_vec_x0)
            hp = torch.cat([hp, char_emb_hp], dim=2)

        if char_vec_x1 != None:
            char_emb_hh = self.char_embedding_forward(char_vec_x1)
            hh = torch.cat([hh, char_emb_hh], dim=2)

        hp, _ = self.prem_gru(hp)
        hh, _ = self.hypo_gru(hh)

        qtc = self.concat_attn(hp, hh)
        qtb = self.bilinear_attn(hp, hh)
        qts = self.dot_attn_1(hp, hh)
        qtd = self.dot_attn_2(hh, hp)
        qtm = self.minus_attn(hp, hh)

        aggregation = torch.cat([hh, qts, qtc, qtd, qtb, qtm], 2)
        aggregation_representation, _ = self.gru_agg(aggregation)
        sj = self.vq(torch.tanh(self.Wq(hp))).transpose(2, 1)
        rq = F.softmax(sj, 2).bmm(hp)
        sj = F.softmax(
            self.vp(self.Wp1(aggregation_representation) + self.Wp2(rq)).transpose(
                2, 1
            ),
            2,
        )
        rp = sj.bmm(aggregation_representation)
        encoder_output = self.dropout(F.leaky_relu(self.prediction(rp)))
        encoder_output = encoder_output.squeeze(1)
        return encoder_output


def mwan_snli(options):
    return MwAN_snli(options)


"""
ESIM
"""


class ESIM(nn.Module):
    def __init__(self, conf):
        super(ESIM, self).__init__()
        self.conf = conf
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

        self.projection = nn.Sequential(
            nn.Linear(4 * 2 * conf["hidden_size"], conf["hidden_size"]), nn.ReLU()
        )
        self.composition = nn.LSTM(
            input_size=conf["hidden_size"],
            hidden_size=conf["hidden_size"],
            num_layers=conf["num_layers"],
            dropout=conf["dropout"],
            bidirectional=True,
            batch_first=True,
        )
        self.classification = nn.Sequential(
            nn.Dropout(p=conf["dropout"]),
            nn.Linear(2 * 4 * conf["hidden_size"], conf["hidden_size"]),
            nn.Tanh(),
            nn.Dropout(p=conf["dropout"]),
            nn.Linear(conf["hidden_size"], 3),
        )

    def forward(self, x0, x1):
        x0_enc = self.encode(x0)
        x1_enc = self.encode(x1)

        x0_att, x1_att = self.softmax_attention(x0_enc, x1_enc)

        enh_x0 = torch.cat([x0_enc, x0_att, x0_enc - x0_att, x0_enc * x0_att], dim=-1)
        enh_x1 = torch.cat([x1_enc, x1_att, x1_enc - x1_att, x1_enc * x1_att], dim=-1)

        proj_x0 = self.dropout(self.projection(enh_x0))
        proj_x1 = self.dropout(self.projection(enh_x1))

        comp_x0, (_, _) = self.composition(proj_x0)
        comp_x1, (_, _) = self.composition(proj_x1)

        avg_x0 = torch.mean(comp_x0, dim=1)
        avg_x1 = torch.mean(comp_x1, dim=1)

        max_x0 = torch.max(comp_x0, dim=1).values
        max_x1 = torch.max(comp_x1, dim=1).values

        v = torch.cat([avg_x0, avg_x1, max_x0, max_x1], dim=1)
        return self.classification(v)

    def softmax_attention(self, x, y):
        similarity_matrix = x.bmm(y.transpose(2, 1).contiguous())
        x_att = F.softmax(similarity_matrix, dim=1)
        y_att = F.softmax(similarity_matrix.transpose(1, 2).contiguous(), dim=1)
        x_att_emb = x_att.bmm(y)
        y_att_emb = y_att.bmm(x)
        return x_att_emb, y_att_emb

    def encode(self, x):
        embedded = self.embedding(x)
        embedded = self.relu(self.translate(embedded))
        all_, (_, _) = self.lstm_layer(embedded)
        return all_


def esim_snli(options):
    return ESIM(options)


