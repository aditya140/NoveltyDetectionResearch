import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from src.model.nli_models import *


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1 or classname.find("Bilinear") != -1:
        nn.init.kaiming_uniform_(
            a=2, mode="fan_in", nonlinearity="leaky_relu", tensor=m.weight
        )
        if m.bias is not None:
            nn.init.constant_(tensor=m.bias, val=0)

    elif classname.find("Conv") != -1:
        nn.init.kaiming_uniform_(
            a=2, mode="fan_in", nonlinearity="leaky_relu", tensor=m.weight
        )
        if m.bias is not None:
            nn.init.constant_(tensor=m.bias, val=0)

    elif (
        classname.find("BatchNorm") != -1
        or classname.find("GroupNorm") != -1
        or classname.find("LayerNorm") != -1
    ):
        nn.init.uniform_(a=0, b=1, tensor=m.weight)
        nn.init.constant_(tensor=m.bias, val=0)

    elif classname.find("Cell") != -1:
        nn.init.xavier_uniform_(gain=1, tensor=m.weight_hh)
        nn.init.xavier_uniform_(gain=1, tensor=m.weight_ih)
        nn.init.ones_(tensor=m.bias_hh)
        nn.init.ones_(tensor=m.bias_ih)

    # elif (
    #     classname.find("RNN") != -1
    #     or classname.find("LSTM") != -1
    #     or classname.find("GRU") != -1
    # ):
    #     for w in m.all_weights:
    #         nn.init.xavier_uniform_(gain=1, tensor=w[2].data)
    #         nn.init.xavier_uniform_(gain=1, tensor=w[3].data)
    #         nn.init.ones_(tensor=w[0].data)
    #         nn.init.ones_(tensor=w[1].data)

    if classname.find("Embedding") != -1:
        nn.init.kaiming_uniform_(
            a=2, mode="fan_in", nonlinearity="leaky_relu", tensor=m.weight
        )


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Decomposable Attention Network
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class DAN(nn.Module):
    def __init__(self, conf, encoder):
        super(DAN, self).__init__()
        self.conf = conf
        self.num_sent = conf["max_num_sent"]
        self.encoder = encoder
        if self.conf["freeze_encoder"]:
            self.encoder.requires_grad_(False)

        self.translate = nn.Linear(
            2 * self.conf["encoder_dim"], self.conf["hidden_size"]
        )
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf["dropout"])

        self.mlp_f = nn.Linear(self.conf["hidden_size"], self.conf["hidden_size"])
        self.mlp_g = nn.Linear(2 * self.conf["hidden_size"], self.conf["hidden_size"])
        self.mlp_h = nn.Linear(2 * self.conf["hidden_size"], self.conf["hidden_size"])
        self.linear = nn.Linear(self.conf["hidden_size"], 2)

    def encode_sent(self, inp):
        batch_size, num_sent, max_len = inp.shape
        x = inp.view(-1, max_len)

        x_padded_idx = x.sum(dim=1) != 0
        x_enc = []
        x_attn = []
        for sub_batch in x[x_padded_idx].split(64):
            # x_enc.append(self.encoder(sub_batch, None))
            x, att = self.encoder(sub_batch, None)
            x_enc.append(x)
            x_attn.append(att)
        x_enc = torch.cat(x_enc, dim=0)
        x_attn = torch.cat(x_attn, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )

        x_attn_t = torch.zeros((batch_size * num_sent, x_attn.size(1), 1)).to(
            self.template.device
        )

        x_enc_t[x_padded_idx] = x_enc
        x_attn_t[x_padded_idx] = x_attn

        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)
        x_attn_t = x_attn_t.view(batch_size, num_sent, -1)
        word_attn = x_attn_t

        embedded = self.dropout(self.translate(x_enc_t))
        embedded = self.act(embedded)
        return embedded, word_attn

    def forward(self, x0, x1):
        x0_enc, x0_att = self.encode_sent(x0)
        x1_enc, x1_att = self.encode_sent(x1)

        f1 = self.act(self.dropout(self.mlp_f(x0_enc)))
        f2 = self.act(self.dropout(self.mlp_f(x1_enc)))

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        prob1 = F.softmax(score1.view(-1, self.num_sent), dim=1).view(
            -1, self.num_sent, self.num_sent
        )

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()

        prob2 = F.softmax(score2.view(-1, self.num_sent), dim=1).view(
            -1, self.num_sent, self.num_sent
        )

        sent1_combine = torch.cat((x0_enc, torch.bmm(prob1, x1_enc)), 2)
        sent2_combine = torch.cat((x1_enc, torch.bmm(prob2, x0_enc)), 2)

        g1 = self.act(self.dropout(self.mlp_g(sent1_combine)))
        g2 = self.act(self.dropout(self.mlp_g(sent2_combine)))

        sent1_output = torch.sum(g1, 1)
        sent1_output = torch.squeeze(sent1_output, 1)

        sent2_output = torch.sum(g2, 1)
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat(
            (sent1_output * sent2_output, torch.abs(sent1_output - sent2_output)), 1
        )

        h = self.act(self.dropout(self.mlp_h(input_combine)))
        opt = self.linear(h)
        return opt


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Asynchronous Deep Interactive Network
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class InferentialModule(nn.Module):
    def __init__(self, conf):
        super(InferentialModule, self).__init__()
        self.W = nn.Linear(conf["hidden_size"], conf["k"], bias=False)
        self.P = nn.Linear(conf["k"], 1, bias=False)
        self.Wb = nn.Linear(4 * conf["hidden_size"], conf["hidden_size"])
        self.LayerNorm = nn.LayerNorm(conf["hidden_size"])

    def forward(self, ha, hb):
        e = F.softmax(self.P(torch.tanh(self.W(ha * hb))))
        hb_d = ha * e
        hb_dd = torch.cat([hb, hb_d, hb - hb_d, hb * hb_d], dim=2)
        hb_b = self.LayerNorm(F.relu(self.Wb(hb_dd)))
        return hb_b


class AsyncInfer(nn.Module):
    def __init__(self, conf):
        super(AsyncInfer, self).__init__()
        self.inf1 = InferentialModule(conf)
        self.inf2 = InferentialModule(conf)
        self.lstm_layer1 = nn.LSTM(
            input_size=int(2 * conf["hidden_size"]),
            hidden_size=int(conf["hidden_size"] / 2),
            num_layers=conf["num_layers"],
            bidirectional=True,
        )
        self.lstm_layer2 = nn.LSTM(
            input_size=int(2 * conf["hidden_size"]),
            hidden_size=int(conf["hidden_size"] / 2),
            num_layers=conf["num_layers"],
            bidirectional=True,
        )

    def forward(self, Vp, Vq):
        vq_hat = self.inf1(Vp, Vq)
        vp_hat = self.inf2(vq_hat, Vp)
        vq_d = torch.cat([Vq, vq_hat], dim=2)
        vp_d = torch.cat([Vp, vp_hat], dim=2)
        Vq_new, (_, _) = self.lstm_layer1(vq_d)
        Vp_new, (_, _) = self.lstm_layer1(vp_d)
        return Vp_new, Vq_new


class ADIN(nn.Module):
    def __init__(self, conf, encoder):
        super(ADIN, self).__init__()
        self.conf = conf
        self.num_sent = conf["max_num_sent"]
        self.encoder = encoder
        if self.conf["freeze_encoder"]:
            self.encoder.requires_grad_(False)

        self.translate = nn.Linear(
            2 * self.conf["encoder_dim"], self.conf["hidden_size"]
        )
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.act = nn.ReLU()
        self.inference_modules = nn.ModuleList(
            [AsyncInfer(conf) for i in range(conf["N"])]
        )
        self.r = nn.Linear(8 * conf["hidden_size"], conf["hidden_size"])
        self.v = nn.Linear(conf["hidden_size"], 2)
        self.dropout = nn.Dropout(p=self.conf["dropout"])

    def encode_sent(self, inp):
        batch_size, num_sent, max_len = inp.shape
        x = inp.view(-1, max_len)

        x_padded_idx = x.sum(dim=1) != 0
        x_enc = []
        x_attn = []
        for sub_batch in x[x_padded_idx].split(64):
            # x_enc.append(self.encoder(sub_batch, None))
            x, att = self.encoder(sub_batch, None)
            x_enc.append(x)
            x_attn.append(att)
        x_enc = torch.cat(x_enc, dim=0)
        x_attn = torch.cat(x_attn, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )

        x_attn_t = torch.zeros((batch_size * num_sent, x_attn.size(1), 1)).to(
            self.template.device
        )

        x_enc_t[x_padded_idx] = x_enc
        x_attn_t[x_padded_idx] = x_attn

        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)
        x_attn_t = x_attn_t.view(batch_size, num_sent, -1)
        word_attn = x_attn_t

        embedded = self.dropout(self.translate(x_enc_t))
        embedded = self.act(embedded)
        return embedded, word_attn

    def forward(self, x0, x1, x0_char_vec=None, x1_char_vec=None):
        x0_enc, x0_att = self.encode_sent(x0)
        x1_enc, x1_att = self.encode_sent(x1)

        for inf_module in self.inference_modules:
            x0_enc, x1_enc = inf_module(x0_enc, x1_enc)

        x0_mean = torch.mean(x0_enc, dim=1)
        x1_mean = torch.mean(x1_enc, dim=1)

        x0_max = torch.max(x0_enc, dim=1)[0]
        x1_max = torch.max(x1_enc, dim=1)[0]

        x0_new = torch.cat([x0_mean, x0_max], dim=1)
        x1_new = torch.cat([x1_mean, x1_max], dim=1)

        r = torch.cat([x0_new, x1_new, x0_new - x1_new, x0_new * x1_new], dim=1)
        v = F.relu(self.r(r))
        y = self.v(v)
        return y


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Hierarchical Attention Network
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


class HAN_DOC(nn.Module):
    def __init__(self, conf, encoder):
        super(HAN_DOC, self).__init__()
        self.conf = conf
        self.encoder = encoder
        if self.conf["freeze_encoder"]:
            self.encoder.requires_grad_(False)

        self.translate = nn.Linear(
            2 * self.conf["encoder_dim"], self.conf["hidden_size"]
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf["dropout"])
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.lstm_layer = nn.LSTM(
            input_size=self.conf["hidden_size"],
            hidden_size=self.conf["hidden_size"],
            num_layers=self.conf["num_layers"],
            bidirectional=True,
        )
        conf["attention_input"] = conf["hidden_size"]
        self.attention = StrucSelfAttention(conf)

    def forward(self, inp):
        batch_size, num_sent, max_len = inp.shape
        x = inp.view(-1, max_len)

        x_padded_idx = x.sum(dim=1) != 0
        x_enc = []
        x_attn = []
        for sub_batch in x[x_padded_idx].split(64):
            # x_enc.append(self.encoder(sub_batch, None))
            x, att = self.encoder(sub_batch, None)
            x_enc.append(x)
            x_attn.append(att)
        x_enc = torch.cat(x_enc, dim=0)
        x_attn = torch.cat(x_attn, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )

        x_attn_t = torch.zeros((batch_size * num_sent, x_attn.size(1), 1)).to(
            self.template.device
        )

        x_enc_t[x_padded_idx] = x_enc
        x_attn_t[x_padded_idx] = x_attn

        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)
        x_attn_t = x_attn_t.view(batch_size, num_sent, -1)
        word_attn = x_attn_t

        embedded = self.dropout(self.translate(x_enc_t))
        embedded = self.act(embedded)

        all_, (_, _) = self.lstm_layer(embedded)
        cont, attn = self.attention(all_, return_attention=True)
        attn = torch.mean(attn,2)
        return cont, attn, word_attn


class HAN_DOC_Classifier(nn.Module):
    def __init__(self, conf, encoder):
        super().__init__()
        self.conf = conf
        self.encoder = HAN_DOC(conf, encoder)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf["dropout"])
        self.fc = nn.Linear(2 * conf["hidden_size"] * conf["attention_hops"], 10)

    def forward(self, x0):
        x0_enc, _, _ = self.encoder(x0)
        x0_enc = x0_enc.flatten(start_dim=1)
        cont = self.dropout(self.act(x0_enc))
        cont = self.fc(cont)
        return cont


class HAN(nn.Module):
    def __init__(self, conf, encoder, doc_enc=None):
        super(HAN, self).__init__()
        self.conf = conf
        if doc_enc == None:
            self.encoder = HAN_DOC(conf, encoder)
        elif encoder == None:
            self.encoder = doc_enc
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf["dropout"])
        self.fc = nn.Linear(8 * conf["attention_hops"] * conf["hidden_size"], 2)

    def forward(self, x0, x1):
        x0_enc, _, _ = self.encoder(x0)
        x1_enc, _, _ = self.encoder(x1)

        cont = torch.cat(
            [
                x0_enc,
                x1_enc,
                torch.abs(x0_enc - x1_enc),
                x0_enc * x1_enc,
            ],
            dim=2,
        )
        cont = cont.flatten(start_dim=1)
        cont = self.dropout(self.act(cont))
        cont = self.fc(cont)
        return cont

    def forward_with_attn(self, x0, x1):
        x0_enc, x0_attn, x0_word_attn = self.encoder(x0)
        x1_enc, x1_attn, x1_word_attn = self.encoder(x1)
        cont = torch.cat(
            [
                x0_enc,
                x1_enc,
                torch.abs(x0_enc - x1_enc),
                x0_enc * x1_enc,
            ],
            dim=2,
        )
        cont = cont.flatten(start_dim=1)
        cont = self.dropout(self.act(cont))
        cont = self.fc(cont)
        return cont, (x0_attn, x0_word_attn), (x1_attn, x1_word_attn)


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Relative Document Vector CNN
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class Accumulator(nn.Module):
    def __init__(self, conf, encoder):
        super(Accumulator, self).__init__()
        self.conf = conf
        self.encoder = encoder
        if self.conf["freeze_encoder"]:
            self.encoder.requires_grad_(False)

        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)

    def forward(self, trg, src):
        batch_size, num_sent, max_len = src.shape

        x = src.view(-1, max_len)
        y = trg.view(-1, max_len)

        x_padded_idx = x.sum(dim=1) != 0
        y_padded_idx = y.sum(dim=1) != 0

        x_enc = []

        for sub_batch in x[x_padded_idx].split(64):
            x_enc.append(self.encoder(sub_batch, None)[0])
        x_enc = torch.cat(x_enc, dim=0)
        y_enc = []

        for sub_batch in y[y_padded_idx].split(64):
            y_enc.append(self.encoder(sub_batch, None)[0])
        y_enc = torch.cat(y_enc, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )
        x_enc_t[x_padded_idx] = x_enc

        y_enc_t = torch.zeros((batch_size * num_sent, y_enc.size(1))).to(
            self.template.device
        )
        y_enc_t[y_padded_idx] = y_enc

        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)
        y_enc_t = y_enc_t.view(batch_size, num_sent, -1)
        eps = 1e-8

        a_n = x_enc_t.norm(dim=2)[:, None]
        a_norm = x_enc_t / torch.max(
            eps * torch.ones_like(a_n.permute(0, 2, 1)), a_n.permute(0, 2, 1)
        )

        b_n = y_enc_t.norm(dim=2)[:, None]
        b_norm = y_enc_t / torch.max(
            eps * torch.ones_like(b_n.permute(0, 2, 1)), b_n.permute(0, 2, 1)
        )
        cos_sim = torch.bmm(a_norm, b_norm.transpose(1, 2))
        y_sim = torch.argmax(cos_sim, dim=2)
        dummy = y_sim.unsqueeze(2).expand(y_sim.size(0), y_sim.size(1), y_enc_t.size(2))
        matched_y = torch.gather(y_enc_t, 1, dummy)

        rdv = torch.cat(
            [
                x_enc_t,
                matched_y,
                torch.abs(x_enc_t - matched_y),
                x_enc_t * matched_y,
            ],
            dim=2,
        )
        return rdv


class RDV_CNN(nn.Module):
    def __init__(self, conf, encoder):
        super(RDV_CNN, self).__init__()
        self.accumulator = Accumulator(conf, encoder)
        if conf["freeze_encoder"]:
            print("Encoder Freezed")
            self.accumulator.requires_grad_(False)
        self.linear = nn.Linear(conf["num_filters"] * len(conf["filter_sizes"]), 2)
        self.convs1 = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    conf["num_filters"],
                    (K, conf["encoder_dim"] * 2 * 4),
                )
                for K in conf["filter_sizes"]
            ]
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf["dropout"])

    def forward(self, x, y):
        rdv = self.accumulator(x, y)
        opt = [self.act(conv(rdv.unsqueeze(1))).squeeze(3) for conv in self.convs1]
        opt = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in opt]
        opt = torch.cat(opt, 1)
        opt = self.act(opt)
        opt = self.linear(opt)
        return opt


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Deep Interactive Inference Network
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class Highway(nn.Module):
    def __init__(self, size, num_layers, dropout):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList(
            [nn.Linear(size, size) for _ in range(num_layers)]
        )

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.f = nn.Tanh()

    def forward(self, x):
        """
        :param x: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
        f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
        and ⨀ is element-wise multiplication
        """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            linear = self.dropout(linear)
            x = gate * nonlinear + (1 - gate) * linear
        return x


class self_attention(nn.Module):
    """[summary]
    Self attention modeled as demonstrated in NATURAL LANGUAGE INFERENCE OVER INTERACTION SPACE paper

    REF: https://github.com/YerevaNN/DIIN-in-Keras/blob/master/layers/encoding.py

    # P = P_hw
    # itr_attn = P_itrAtt
    # encoding = P_enc
    # The paper takes inputs to be P(_hw) as an example and then computes the same thing for H,
    # therefore we'll name our inputs P too.

    # Input of encoding is P with shape (batch, p, d). It would be (batch, h, d) for hypothesis
    # Construct alphaP of shape (batch, p, 3*d, p)
    # A = dot(w_itr_att, alphaP)

    # alphaP consists of 3*d rows along 2nd axis
    # 1. up   -> first  d items represent P[i]
    # 2. mid  -> second d items represent P[j]
    # 3. down -> final items represent alpha(P[i], P[j]) which is element-wise product of P[i] and P[j] = P[i]*P[j]

    # If we look at one slice of alphaP we'll see that it has the following elements:
    # ----------------------------------------
    # P[i][0], P[i][0], P[i][0], ... P[i][0]   ▲
    # P[i][1], P[i][1], P[i][1], ... P[i][1]   |
    # P[i][2], P[i][2], P[i][2], ... P[i][2]   |
    # ...                              ...     | up
    #      ...                         ...     |
    #             ...                  ...     |
    # P[i][d], P[i][d], P[i][d], ... P[i][d]   ▼
    # ----------------------------------------
    # P[0][0], P[1][0], P[2][0], ... P[p][0]   ▲
    # P[0][1], P[1][1], P[2][1], ... P[p][1]   |
    # P[0][2], P[1][2], P[2][2], ... P[p][2]   |
    # ...                              ...     | mid
    #      ...                         ...     |
    #             ...                  ...     |
    # P[0][d], P[1][d], P[2][d], ... P[p][d]   ▼
    # ----------------------------------------
    #                                          ▲
    #                                          |
    #                                          |
    #               up * mid                   | down
    #          element-wise product            |
    #                                          |
    #                                          ▼
    # ----------------------------------------

    # For every slice(i) the up part changes its P[i] values
    # The middle part is repeated p times in depth (for every i)
    # So we can get the middle part by doing the following:
    # mid = broadcast(P) -> to get tensor of shape (batch, p, d, p)
    # As we can notice up is the same mid, but with changed axis, so to obtain up from mid we can do:
    # up = swap_axes(mid, axis1=0, axis2=2)

    # P_itr_attn[i] = sum of for j = 1...p:
    #                           s = sum(for k = 1...p:  e^A[k][j]
    #                           ( e^A[i][j] / s ) * P[j]  --> P[j] is the j-th row, while the first part is a number
    # So P_itr_attn is the weighted sum of P
    # SA is column-wise soft-max applied on A
    # P_itr_attn[i] is the sum of all rows of P scaled by i-th row of SA

    """

    def __init__(self, conf):
        super().__init__()
        self.w = nn.Linear(3 * conf["hidden_size"], 1, bias=False)
        self.dropout = nn.Dropout(conf["dropout"][2])

    def forward(self, p):
        # p = [B,P,D]
        p_dim = p.shape[1]
        mid = p.unsqueeze(3).expand(-1, -1, -1, p_dim)
        # min = [B,P,D,P]
        up = mid.permute(0, 3, 2, 1)
        alpha = torch.cat([up, mid, up * mid], dim=2)
        A = (self.w.weight @ alpha).squeeze(2)
        A = self.dropout(A)
        sA = A.softmax(dim=2)
        itr_attn = torch.bmm(sA, p)
        return itr_attn


class fuse_gate(nn.Module):
    """[summary]
    Fuse gate is used to provide a Skip connection for the encoding and the attended output.
    The author uses:
    zi = tanh(W1 * [P:P_att]+b1)
    ri = Sigmoid(W2 * [P:P_att]+b2)
    fi = Sigmoid(W3 * [P:P_att]+b3)
    P_new  = r dot P + fi dot zi

    W1,W2,W3 = Linear(2d,d)

    """

    def __init__(self, conf):
        super().__init__()
        self.fc1 = nn.Linear(conf["hidden_size"] * 2, conf["hidden_size"])
        self.fc2 = nn.Linear(conf["hidden_size"] * 2, conf["hidden_size"])
        self.fc3 = nn.Linear(conf["hidden_size"] * 2, conf["hidden_size"])
        self.dropout = nn.Dropout(conf["dropout"][3])

    def forward(self, p_hat_i, p_dash_i):
        x = torch.cat([p_hat_i, p_dash_i], dim=2)
        z = torch.tanh(self.dropout(self.fc1(x)))
        r = torch.sigmoid(self.dropout(self.fc1(x)))
        f = torch.sigmoid(self.dropout(self.fc1(x)))
        enc = r * p_hat_i + f * z
        return enc


class interaction(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, p, h):
        p = p.unsqueeze(2)
        h = h.unsqueeze(1)
        return p * h


class Dense_net_block(nn.Module):
    def __init__(self, outChannels, growth_rate, kernel_size):
        super(Dense_net_block, self).__init__()
        self.conv = nn.Conv2d(
            outChannels, growth_rate, kernel_size=kernel_size, bias=False, padding=1
        )

    def forward(self, x):
        ft = F.relu(self.conv(x))
        out = torch.cat((x, ft), dim=1)
        return out


class Dense_net_transition(nn.Module):
    def __init__(self, nChannels, outChannels):
        super(Dense_net_transition, self).__init__()
        self.conv = nn.Conv2d(nChannels, outChannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(x)
        out = F.max_pool2d(out, (2, 2), (2, 2), padding=0)
        return out


class DenseNet(nn.Module):
    def __init__(self, nChannels, growthRate, reduction, nDenseBlocks, kernel_size):
        super(DenseNet, self).__init__()
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, kernel_size)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Dense_net_transition(nChannels, nOutChannels)
        nChannels = nOutChannels

        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, kernel_size)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Dense_net_transition(nChannels, nOutChannels)
        nChannels = nOutChannels

        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, kernel_size)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans3 = Dense_net_transition(nChannels, nOutChannels)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, kernel_size):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(Dense_net_block(nChannels, growthRate, kernel_size))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = self.trans1(self.dense1(x))
        # print(out.shape)
        out = self.trans2(self.dense2(out))
        # print(out.shape)
        out = self.trans3(self.dense3(out))
        # print(out.shape)
        return out


class DIIN(nn.Module):
    def __init__(self, conf, encoder):
        super().__init__()
        self.conf = conf
        self.encoder = encoder
        if self.conf["freeze_encoder"]:
            self.encoder.requires_grad_(False)

        self.num_sent = conf["max_num_sent"]

        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.dropout = nn.Dropout(conf["dropout"][4])

        self.translate = nn.Linear(2 * conf["hidden_size"], self.conf["hidden_size"])
        self.highway = Highway(
            self.conf["hidden_size"], conf["num_layers"], conf["dropout"][1]
        )
        self.attn = self_attention(self.conf)
        self.fuse = fuse_gate(self.conf)
        self.interact = interaction(self.conf)
        self.interaction_cnn = nn.Conv2d(
            self.conf["hidden_size"],
            int(
                self.conf["hidden_size"] * self.conf["dense_net_first_scale_down_ratio"]
            ),
            self.conf["first_scale_down_kernel"],
            padding=0,
        )
        nChannels = int(
            self.conf["hidden_size"] * self.conf["dense_net_first_scale_down_ratio"]
        )

        features = self.num_sent

        for i in range(3):
            nChannels += (
                self.conf["dense_net_layers"] * self.conf["dense_net_growth_rate"]
            )
            nOutChannels = int(
                math.floor(nChannels * self.conf["dense_net_transition_rate"])
            )
            nChannels = nOutChannels
            features = features // 2
        final_layer_size = (features ** 2) * nChannels

        self.dense_net = DenseNet(
            int(
                self.conf["hidden_size"] * self.conf["dense_net_first_scale_down_ratio"]
            ),
            self.conf["dense_net_growth_rate"],
            self.conf["dense_net_transition_rate"],
            self.conf["dense_net_layers"],
            self.conf["dense_net_kernel_size"],
        )
        self.fc1 = nn.Linear(final_layer_size, 2)
        self.act = nn.ReLU()

    def encoder_sent(self, inp):
        batch_size, num_sent, max_len = inp.shape
        x = inp.view(-1, max_len)

        x_padded_idx = x.sum(dim=1) != 0
        x_enc = []
        x_attn = []
        for sub_batch in x[x_padded_idx].split(64):
            # x_enc.append(self.encoder(sub_batch, None))
            x, att = self.encoder(sub_batch, None)
            x_enc.append(x)
            x_attn.append(att)
        x_enc = torch.cat(x_enc, dim=0)
        x_attn = torch.cat(x_attn, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )

        x_attn_t = torch.zeros((batch_size * num_sent, x_attn.size(1), 1)).to(
            self.template.device
        )

        x_enc_t[x_padded_idx] = x_enc
        x_attn_t[x_padded_idx] = x_attn

        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)
        x_attn_t = x_attn_t.view(batch_size, num_sent, -1)
        word_attn = x_attn_t

        embedded = self.dropout(self.translate(x_enc_t))
        embedded = self.act(embedded)
        return embedded

    def encode_attn(self, x):
        x = self.encoder_sent(x)
        x = self.highway(x)
        x_att = self.attn(x)
        enc = self.fuse(x, x_att)
        return enc

    def forward(self, source, target):
        batch_size = source.shape[0]
        s_enc = self.encode_attn(source)
        # print(p_enc.shape)
        t_enc = self.encode_attn(target)
        intr = self.interact(s_enc, t_enc).permute(0, 3, 1, 2)
        # print(intr.shape)
        fm = self.interaction_cnn(intr)
        # print(fm.shape)
        dense = self.dense_net(fm)
        # print(dense.shape)
        opt = self.fc1(dense.reshape(batch_size, -1))
        # print(opt.shape)
        return opt


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


class MwAN(nn.Module):
    def __init__(self, conf, encoder):
        super().__init__()
        self.conf = conf
        self.dropout = nn.Dropout(conf["dropout"])
        self.encoder = encoder
        if self.conf["freeze_encoder"]:
            self.encoder.requires_grad_(False)

        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)

        self.translate = nn.Linear(
            2 * self.conf["encoder_dim"], 2 * self.conf["hidden_size"]
        )

        self.prem_gru = nn.GRU(
            input_size=2 * self.conf["encoder_dim"],
            hidden_size=self.conf["hidden_size"],
            batch_first=True,
            bidirectional=True,
        )

        self.hypo_gru = nn.GRU(
            input_size=2 * self.conf["encoder_dim"],
            hidden_size=self.conf["hidden_size"],
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

        self.prediction = nn.Linear(2 * conf["hidden_size"], 2, bias=True)
        self.initialize()

    def initialize(self):
        print("Initialized weights")
        dont_init = self.encoder.__class__.__name__
        for i in self.modules():
            if i.__class__.__name__ != dont_init:
                i.apply(init_weights)

    def encode_sent(self, inp):
        batch_size, num_sent, max_len = inp.shape
        x = inp.view(-1, max_len)

        x_padded_idx = x.sum(dim=1) != 0
        x_enc = []
        x_attn = []
        for sub_batch in x[x_padded_idx].split(64):
            # x_enc.append(self.encoder(sub_batch, None))
            x, att = self.encoder(sub_batch, None)
            x_enc.append(x)
            x_attn.append(att)
        x_enc = torch.cat(x_enc, dim=0)
        x_attn = torch.cat(x_attn, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )

        x_attn_t = torch.zeros((batch_size * num_sent, x_attn.size(1), 1)).to(
            self.template.device
        )

        x_enc_t[x_padded_idx] = x_enc
        x_attn_t[x_padded_idx] = x_attn

        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)
        x_attn_t = x_attn_t.view(batch_size, num_sent, -1)
        # embedded = self.dropout(self.translate(x_enc_t))

        return x_enc_t

    def forward(self, x0, x1):
        x1, x0 = x0, x1
        x0_enc = self.encode_sent(x0)
        x1_enc = self.encode_sent(x1)

        x0_enc, _ = self.prem_gru(x0_enc)
        x1_enc, _ = self.hypo_gru(x1_enc)

        qtc = self.concat_attn(x0_enc, x1_enc)
        qtb = self.bilinear_attn(x0_enc, x1_enc)
        qts = self.dot_attn_1(x0_enc, x1_enc)
        qtd = self.dot_attn_2(x1_enc, x0_enc)
        qtm = self.minus_attn(x0_enc, x1_enc)

        aggregation = torch.cat([x1_enc, qts, qtc, qtd, qtb, qtm], 2)
        aggregation_representation, _ = self.gru_agg(aggregation)

        sj = self.vq(torch.tanh(self.Wq(x0_enc))).transpose(2, 1)
        rq = F.softmax(sj, 2).bmm(x0_enc)
        sj = F.softmax(
            self.vp(self.Wp1(aggregation_representation) + self.Wp2(rq)).transpose(
                2, 1
            ),
            2,
        )

        rp = sj.bmm(aggregation_representation)
        encoder_output = self.dropout(F.relu(self.prediction(rp)))
        encoder_output = encoder_output.squeeze(1)
        return encoder_output


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Structured Self Attention + pruning
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class StrucSelfAttention(nn.Module):
    def __init__(self, conf):
        super(StrucSelfAttention, self).__init__()
        self.ut_dense = nn.Linear(
            2 * conf["attention_input"], conf["attention_layer_param"], bias=False
        )
        self.et_dense = nn.Linear(
            conf["attention_layer_param"], conf["attention_hops"], bias=False
        )

    def forward(self, x, return_attention=False):
        # x shape: [batch_size, num_sent, embedding_width]
        # ut shape: [batch_size, num_sent, att_unit]
        ut = self.ut_dense(x)
        ut = torch.tanh(ut)
        # et shape: [batch_size, num_sent, att_hops]
        et = self.et_dense(ut)
        # att shape: [batch_size,  att_hops, seq_len]
        att = F.softmax(et,dim=1)
        # output shape [batch_size, att_hops, embedding_width]
        output = torch.bmm(att.permute(0, 2, 1), x).squeeze(1)
        if return_attention:
            return output, att
        else:
            return output


class Struc_DOC(nn.Module):
    def __init__(self, conf, encoder):
        super(Struc_DOC, self).__init__()
        self.conf = conf
        self.encoder = encoder
        if self.conf["freeze_encoder"]:
            self.encoder.requires_grad_(False)

        self.translate = nn.Linear(
            2 * self.conf["encoder_dim"], self.conf["hidden_size"]
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf["dropout"])
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.lstm_layer = nn.LSTM(
            input_size=self.conf["hidden_size"],
            hidden_size=self.conf["hidden_size"],
            num_layers=self.conf["num_layers"],
            bidirectional=True,
        )
        self.attention = StrucSelfAttention(conf)

        self.prune_p = nn.Linear(2 * self.conf["hidden_size"], self.conf["prune_p"])
        self.prune_q = nn.Linear(self.conf["attention_hops"], self.conf["prune_q"])

    def forward(self, inp):
        batch_size, num_sent, max_len = inp.shape
        x = inp.view(-1, max_len)

        x_padded_idx = x.sum(dim=1) != 0
        x_enc = []
        x_attn = []
        for sub_batch in x[x_padded_idx].split(64):
            # x_enc.append(self.encoder(sub_batch, None))
            x, att = self.encoder(sub_batch, None)
            x_enc.append(x)
            x_attn.append(att)
        x_enc = torch.cat(x_enc, dim=0)
        x_attn = torch.cat(x_attn, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )

        x_attn_t = torch.zeros((batch_size * num_sent, x_attn.size(1), 1)).to(
            self.template.device
        )

        x_enc_t[x_padded_idx] = x_enc
        x_attn_t[x_padded_idx] = x_attn

        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)
        x_attn_t = x_attn_t.view(batch_size, num_sent, -1)
        word_attn = x_attn_t

        embedded = self.dropout(self.translate(x_enc_t))
        embedded = self.act(embedded)

        all_, (_, _) = self.lstm_layer(embedded)
        # opt: [batch, att_hops, hidden_size]
        opt, attn = self.attention(all_)
        # p_section: [batch, att_hops, prune_p]
        p_section = self.prune_p(opt)
        # q_section: [batch, hidden_size, prune_q]
        q_section = self.prune_q(opt.permute(0, 2, 1))
        encoded = torch.cat(
            [p_section.view(batch_size, -1), q_section.view(batch_size, -1)], dim=1
        )
        return encoded


class StrucSelfAttn(nn.Module):
    def __init__(self, conf, encoder, doc_enc=None):
        super(StrucSelfAttn, self).__init__()
        self.conf = conf
        if doc_enc == None:
            self.encoder = Struc_DOC(conf, encoder)
        elif encoder == None:
            self.encoder = doc_enc
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf["dropout"])

        fc_in_dim = (
            self.conf["attention_hops"] * self.conf["prune_p"]
            + 2 * self.conf["hidden_size"] * self.conf["prune_q"]
        )

        self.fc = nn.Linear(4 * fc_in_dim, 2)

    def forward(self, x0, x1):
        # x0, x1 = inputs
        x0_enc = self.encoder(x0)
        x1_enc = self.encoder(x1)

        cont = torch.cat(
            [
                x0_enc,
                x1_enc,
                torch.abs(x0_enc - x1_enc),
                x0_enc * x1_enc,
            ],
            dim=1,
        )
        cont = self.dropout(self.act(cont))
        cont = self.fc(cont)
        return cont


"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Multi Attention Model 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

### Attention Functions copied from MwAN


class aggregation_layer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wq = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.vq = nn.Linear(hidden_size, 1, bias=False)

        self.Wp1 = nn.Linear(4 * hidden_size, hidden_size, bias=False)
        self.Wp2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.vp = nn.Linear(hidden_size, 1, bias=False)

        self.prediction = nn.Linear(4 * hidden_size, 2, bias=True)

    def forward(self, x0_enc, agg_rep):
        # print(self.Wq(x0_enc).shape)
        self.dropout = nn.Dropout(0.2)
        sj = self.vq(torch.tanh(self.Wq(x0_enc))).transpose(2, 1)
        rq = F.softmax(sj, 2).bmm(x0_enc)
        sj = F.softmax(
            self.vp(self.Wp1(agg_rep) + self.Wp2(rq)).transpose(2, 1),
            2,
        )
        rp = sj.bmm(agg_rep)
        encoder_output = self.dropout(F.relu(self.prediction(rp)))
        encoder_output = encoder_output.squeeze(1)
        return encoder_output


class attention_fx(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.att_type = conf["attention_type"]

        if conf["attention_type"] == "dot":
            self.attention = dot_attention(conf["hidden_size"])
        elif conf["attention_type"] == "minus":
            self.attention = minus_attention(conf["hidden_size"])
        elif conf["attention_type"] == "concat":
            self.attention = concat_attention(conf["hidden_size"])
        elif conf["attention_type"] == "bilinear_attention":
            self.attention = bilinear_attention(conf["hidden_size"])
        elif conf["attention_type"] == "all":
            self.concat_attn = concat_attention(conf["hidden_size"])
            self.bilinear_attn = bilinear_attention(conf["hidden_size"])
            self.dot_attn_1 = dot_attention(conf["hidden_size"])
            self.dot_attn_2 = dot_attention(conf["hidden_size"])
            self.minus_attn = minus_attention(conf["hidden_size"])

    def forward(self, x0_enc, x1_enc):
        if self.att_type == "all":
            qtc = self.concat_attn(x0_enc, x1_enc)
            qtb = self.bilinear_attn(x0_enc, x1_enc)
            qts = self.dot_attn_1(x0_enc, x1_enc)
            qtd = self.dot_attn_2(x1_enc, x0_enc)
            qtm = self.minus_attn(x0_enc, x1_enc)
            cont = torch.cat([x1_enc, qts, qtc, qtd, qtb, qtm], 2)

            # maxpool agg
            agg_res = torch.max(
                torch.cat(
                    [i.unsqueeze(0) for i in [x1_enc, qts, qtc, qtd, qtb, qtm]], dim=0
                ),
                dim=0,
            ).values
            return agg_res
        else:
            att_res = self.attention(x0_enc, x1_enc)
            return att_res


class MultiAtt(nn.Module):
    def __init__(self, conf, encoder):
        super(MultiAtt, self).__init__()
        self.conf = conf
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf["dropout"])
        self.encoder = doc_encoder(conf, encoder)
        self.attention_fx = attention_fx(conf)
        self.aggregate = aggregation_layer(conf["hidden_size"])

    def forward(self, x0, x1):
        x0_enc = self.encoder(x0)
        x1_enc = self.encoder(x1)

        att_enc = self.attention_fx(x0_enc, x1_enc)
        att_agg = self.dropout(att_enc)
        agg_rep = torch.cat([x1_enc, att_agg], dim=2)
        opt = self.aggregate(x0_enc, agg_rep)
        return opt


"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
EIN
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class doc_encoder(nn.Module):
    def __init__(self, conf, encoder):
        super(doc_encoder, self).__init__()
        self.conf = conf
        self.encoder = encoder
        if self.conf["freeze_encoder"]:
            self.encoder.requires_grad_(False)

        self.translate = nn.Linear(
            2 * self.conf["encoder_dim"], self.conf["hidden_size"]
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf["dropout"])
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.lstm_layer = nn.LSTM(
            input_size=self.conf["hidden_size"],
            hidden_size=self.conf["hidden_size"],
            num_layers=self.conf["num_layers"],
            bidirectional=True,
        )

    def forward(self, inp):
        batch_size, num_sent, max_len = inp.shape
        x = inp.view(-1, max_len)

        x_padded_idx = x.sum(dim=1) != 0
        x_enc = []
        x_attn = []
        for sub_batch in x[x_padded_idx].split(64):
            # x_enc.append(self.encoder(sub_batch, None))
            x, att = self.encoder(sub_batch, None)
            x_enc.append(x)
            x_attn.append(att)
        x_enc = torch.cat(x_enc, dim=0)
        x_attn = torch.cat(x_attn, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )

        x_attn_t = torch.zeros((batch_size * num_sent, x_attn.size(1), 1)).to(
            self.template.device
        )

        x_enc_t[x_padded_idx] = x_enc
        x_attn_t[x_padded_idx] = x_attn

        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)
        x_attn_t = x_attn_t.view(batch_size, num_sent, -1)
        word_attn = x_attn_t

        embedded = self.dropout(self.translate(x_enc_t))
        embedded = self.act(embedded)

        all_, (_, _) = self.lstm_layer(embedded)

        return all_


class EIN(nn.Module):
    def __init__(self, conf, encoder):
        super(EIN, self).__init__()
        self.conf = conf
        self.encoder = doc_encoder(conf, encoder)
        self.dropout = nn.Dropout(conf["dropout"])

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
            nn.Linear(conf["hidden_size"], 2),
        )

    def forward(self, x0, x1):
        x0_enc = self.encoder(x0)
        x1_enc = self.encoder(x1)

        x0_att, x1_att = self.softmax_attention(x0_enc, x1_enc)

        enh_x0 = torch.cat(
            [x0_enc, x0_att, torch.abs(x0_enc - x0_att), x0_enc * x0_att], dim=-1
        )
        enh_x1 = torch.cat(
            [x1_enc, x1_att, torch.abs(x1_enc - x1_att), x1_enc * x1_att], dim=-1
        )

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


"""
EIN + self attention
"""


class EAtIn(nn.Module):
    def __init__(self, conf, encoder):
        super(EAtIn, self).__init__()
        self.conf = conf
        self.encoder = doc_encoder(conf, encoder)
        self.dropout = nn.Dropout(conf["dropout"])

        self.attention = Attention(conf)
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
            nn.Linear(2 * 6 * conf["hidden_size"], conf["hidden_size"]),
            nn.Tanh(),
            nn.Dropout(p=conf["dropout"]),
            nn.Linear(conf["hidden_size"], 2),
        )

    def forward(self, x0, x1):
        x0_enc = self.encoder(x0)
        x1_enc = self.encoder(x1)

        x0_self = self.self_attention(x0_enc)
        x1_self = self.self_attention(x1_enc)

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

        i = torch.cat([x0_self, avg_x0, avg_x1, max_x0, max_x1, x1_self], dim=1)
        return self.classification(i)

    def softmax_attention(self, x, y):
        similarity_matrix = x.bmm(y.transpose(2, 1).contiguous())
        x_att = F.softmax(similarity_matrix, dim=1)
        y_att = F.softmax(similarity_matrix.transpose(1, 2).contiguous(), dim=1)
        x_att_emb = x_att.bmm(y)
        y_att_emb = y_att.bmm(x)
        return x_att_emb, y_att_emb

    def self_attention(self, x):
        attn = self.attention(x)
        cont = torch.bmm(attn.permute(0, 2, 1), x)
        cont = cont.squeeze(1)
        return cont

    def encode(self, x):
        embedded = self.embedding(x)
        embedded = self.relu(self.translate(embedded))
        all_, (_, _) = self.lstm_layer(embedded)
        return all_


"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
HAN_ABLATE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class HAN_DOC_ablate(nn.Module):
    def __init__(self, conf, encoder):
        super(HAN_DOC_ablate, self).__init__()
        self.conf = conf
        self.encoder = encoder
        if self.conf["freeze_encoder"]:
            self.encoder.requires_grad_(False)

        self.translate = nn.Linear(
            2 * self.conf["encoder_dim"], self.conf["hidden_size"]
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf["dropout"])
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        if conf["use_bilstm"]:
            self.lstm_layer = nn.LSTM(
                input_size=self.conf["hidden_size"],
                hidden_size=self.conf["hidden_size"],
                num_layers=self.conf["num_layers"],
                bidirectional=True,
            )
            conf["attention_input"] = self.conf["hidden_size"]
        else:
            conf["attention_input"] = self.conf["hidden_size"] // 2

        if self.conf["attention_type"] == "struc":
            self.attention = StrucSelfAttention(conf)

    def forward(self, inp):
        batch_size, num_sent, max_len = inp.shape
        x = inp.view(-1, max_len)

        x_padded_idx = x.sum(dim=1) != 0
        x_enc = []
        x_attn = []
        for sub_batch in x[x_padded_idx].split(64):
            # x_enc.append(self.encoder(sub_batch, None))
            x, att = self.encoder(sub_batch, None)
            x_enc.append(x)
            x_attn.append(att)
        x_enc = torch.cat(x_enc, dim=0)
        x_attn = torch.cat(x_attn, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )

        x_attn_t = torch.zeros((batch_size * num_sent, x_attn.size(1), 1)).to(
            self.template.device
        )

        x_enc_t[x_padded_idx] = x_enc
        x_attn_t[x_padded_idx] = x_attn

        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)
        x_attn_t = x_attn_t.view(batch_size, num_sent, -1)
        word_attn = x_attn_t

        embedded = self.dropout(self.translate(x_enc_t))
        embedded = self.act(embedded)

        if self.conf["use_bilstm"]:
            all_, (_, _) = self.lstm_layer(embedded)
        else:
            all_ = embedded
        opt = self.attention(all_)
        return opt


class HAN_ablate(nn.Module):
    def __init__(self, conf, encoder, doc_enc=None):
        super(HAN_ablate, self).__init__()
        self.conf = conf
        if doc_enc == None:
            print("----- no doc encoder-------")
            self.encoder = HAN_DOC_ablate(conf, encoder)
        elif encoder == None:
            print("-----doc encoder-------")
            self.encoder = doc_enc
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf["dropout"])

        if conf["attention_type"] == "struc":
            fc_in_dim = (
                (
                    conf["attention_hops"]
                    if conf["agg"] == "flatten"
                    else 2
                    if conf["agg"] == "avgmax"
                    else 1
                )
                * (2 if conf["use_bilstm"] else 1)
                * conf["hidden_size"]
            )

        if conf["interaction_type"] == "concat":
            fc_in_dim = fc_in_dim * 2
        if conf["interaction_type"] == "all":
            fc_in_dim = fc_in_dim * 4

        self.fc = nn.Linear(fc_in_dim, 2)

    def forward(self, x0, x1):
        x0_enc = self.encoder(x0)
        x1_enc = self.encoder(x1)
        if self.conf["interaction_type"] == "concat":
            cont = torch.cat(
                [
                    x0_enc,
                    x1_enc,
                ],
                dim=2,
            )

        elif self.conf["interaction_type"] == "all":
            cont = torch.cat(
                [
                    x0_enc,
                    x1_enc,
                    torch.abs(x0_enc - x1_enc),
                    x0_enc * x1_enc,
                ],
                dim=2,
            )

        if self.conf["agg"] == "flatten":
            cont = cont.flatten(start_dim=1)
        elif self.conf["agg"] == "avgmax":
            cont_avg = torch.max(cont, dim=1).values
            cont_max = torch.mean(cont, dim=1)
            cont = torch.cat([cont_avg, cont_max], dim=1)
        elif self.conf["agg"] == "avg":
            cont = torch.mean(cont, dim=1)
        elif self.conf["agg"] == "max":
            cont = torch.mean(cont, dim=1)

        cont = self.dropout(self.act(cont))
        cont = self.fc(cont)
        return cont
