import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from src.model.nli_models import *


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
        for sub_batch in x[x_padded_idx].split(64):
            x_enc.append(self.encoder(sub_batch, None))
        x_enc = torch.cat(x_enc, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )

        x_enc_t[x_padded_idx] = x_enc
        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)

        embedded = self.dropout(self.translate(x_enc_t))
        embedded = self.act(embedded)
        return embedded

    def forward(self, x0, x1):
        x0_enc = self.encode_sent(x0)
        x1_enc = self.encode_sent(x1)

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
        for sub_batch in x[x_padded_idx].split(64):
            x_enc.append(self.encoder(sub_batch, None))
        x_enc = torch.cat(x_enc, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )

        x_enc_t[x_padded_idx] = x_enc
        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)

        embedded = self.dropout(self.translate(x_enc_t))
        embedded = self.act(embedded)
        return embedded

    def forward(self, x0, x1, x0_char_vec=None, x1_char_vec=None):
        x0_enc = self.encode_sent(x0)
        x1_enc = self.encode_sent(x1)

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
        self.attention = Attention(conf)

    def forward(self, inp):
        batch_size, num_sent, max_len = inp.shape
        x = inp.view(-1, max_len)

        x_padded_idx = x.sum(dim=1) != 0
        x_enc = []
        for sub_batch in x[x_padded_idx].split(64):
            x_enc.append(self.encoder(sub_batch, None))
        x_enc = torch.cat(x_enc, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )

        x_enc_t[x_padded_idx] = x_enc
        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)

        embedded = self.dropout(self.translate(x_enc_t))
        embedded = self.act(embedded)

        all_, (_, _) = self.lstm_layer(embedded)
        attn = self.attention(all_)

        cont = torch.bmm(attn.permute(0, 2, 1), all_)
        cont = cont.squeeze(1)
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
        self.fc = nn.Linear(8 * conf["hidden_size"], 2)

    def forward(self, x0, x1):
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
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Relative Document Vector CNN
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class Accumulator(nn.Module):
    def __init__(self, conf, encoder):
        super(Accumulator, self).__init__()
        self.conf = conf
        self.encoder = encoder
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)

    def forward(self, src, trg):
        batch_size, num_sent, max_len = src.shape

        x = src.view(-1, max_len)
        y = trg.view(-1, max_len)

        x_padded_idx = x.sum(dim=1) != 0
        y_padded_idx = y.sum(dim=1) != 0

        x_enc = []

        for sub_batch in x[x_padded_idx].split(64):
            x_enc.append(self.encoder(sub_batch, None))
        x_enc = torch.cat(x_enc, dim=0)
        y_enc = []

        for sub_batch in y[y_padded_idx].split(64):
            y_enc.append(self.encoder(sub_batch, None))
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
        for sub_batch in x[x_padded_idx].split(64):
            x_enc.append(self.encoder(sub_batch, None))
        x_enc = torch.cat(x_enc, dim=0)

        x_enc_t = torch.zeros((batch_size * num_sent, x_enc.size(1))).to(
            self.template.device
        )

        x_enc_t[x_padded_idx] = x_enc
        x_enc_t = x_enc_t.view(batch_size, num_sent, -1)

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
