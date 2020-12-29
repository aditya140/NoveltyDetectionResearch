import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# TODO : Add Embedding penalty


class DIIN_conf:
    embedding_dim = 800
    hidden_size = 300
    num_layers = 2
    freeze_encoder = True
    dense_net_growth_rate = 20
    dense_net_layers = 3
    dense_net_transition_rate = 0.2
    dense_net_kernel_size = 3                       #do not change
    dense_net_channels = 100
    dense_net_first_scale_down_ratio = 0.3
    first_scale_down_kernel = 1                    # [1-5]
    batch_size = 32
    dropout = [0.3,0.3,0.3,0.3,0.3]

    def __init__(self, num_sent, encoder, **kwargs):
        self.encoder = encoder
        self.num_sent = num_sent
        for k, v in kwargs.items():
            setattr(self, k, v)

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
        self.w = nn.Linear(3 * conf.hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(conf.dropout[2])

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
        self.fc1 = nn.Linear(conf.hidden_size * 2, conf.hidden_size)
        self.fc2 = nn.Linear(conf.hidden_size * 2, conf.hidden_size)
        self.fc3 = nn.Linear(conf.hidden_size * 2, conf.hidden_size)
        self.dropout = nn.Dropout(conf.dropout[3])

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
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.encoder = self.conf.encoder
        self.encoder.requires_grad = conf.freeze_encoder
        self.num_sent = conf.num_sent

        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.dropout = nn.Dropout(conf.dropout[4])

        self.translate = nn.Linear(conf.embedding_dim, self.conf.hidden_size)
        # self.highway = Highway(self.conf.hidden_size, conf.num_layers,conf.dropout[1])
        self.attn = self_attention(self.conf)
        # self.fuse = fuse_gate(self.conf)
        self.interact = interaction(self.conf)
        self.interaction_cnn = nn.Conv2d(
            self.conf.hidden_size,
            int(self.conf.hidden_size * self.conf.dense_net_first_scale_down_ratio),
            self.conf.first_scale_down_kernel,
            padding=0,
        )
        nChannels = int(self.conf.hidden_size * self.conf.dense_net_first_scale_down_ratio)
        features = self.num_sent
        for i in range(3):
            nChannels += self.conf.dense_net_layers * self.conf.dense_net_growth_rate
            nOutChannels = int(math.floor(nChannels * self.conf.dense_net_transition_rate))
            nChannels = nOutChannels
            features = features//2
        final_layer_size = ((features**2)*nChannels)

        self.dense_net = DenseNet(
            int(self.conf.hidden_size * self.conf.dense_net_first_scale_down_ratio),
            self.conf.dense_net_growth_rate,
            self.conf.dense_net_transition_rate,
            self.conf.dense_net_layers,
            self.conf.dense_net_kernel_size,
        )
        self.fc1 = nn.Linear(final_layer_size, 2)

        

    def encode(self, src):
        batch_size = src.shape[0]
        x = src.view(-1, self.num_sent)
        x_padded_idx = x.sum(dim=1) != 0
        x_enc = []
        for sub_batch in x[x_padded_idx].split(64):
            x_enc.append(self.encoder(sub_batch)[0])
        x_enc = torch.cat(x_enc, dim=0)
        x_enc_t = torch.zeros((batch_size * self.num_sent, x_enc.size(1))).to(
            self.template.device
        )
        x_enc_t[x_padded_idx] = x_enc
        x_enc_t = x_enc_t.view(batch_size, self.num_sent, -1)
        return x_enc_t

    def encode_attn(self, x):
        # print(x.shape)
        x = self.encode(x)
        # print(x.shape)
        x = self.translate(x)
        # print(x.shape)
        # x = self.highway(x)
        # print(x.shape)
        x_att = self.attn(x)
        # print(x_att.shape)
        # enc = self.fuse(x, x_att)
        # print(enc.shape)
        return x_att

    def forward(self, p_tok, h_tok):
        batch_size = p_tok.shape[0]
        p_enc = self.encode_attn(p_tok)
        # print(p_enc.shape)
        h_enc = self.encode_attn(h_tok)
        intr = self.interact(p_enc, h_enc).permute(0, 3, 1, 2)
        # print(intr.shape)
        fm = self.interaction_cnn(intr)
        # print(fm.shape)
        dense = self.dense_net(fm)
        # print(dense.shape)
        opt = self.fc1(dense.reshape(batch_size, -1))
        # print(opt.shape)
        return opt

# from utils import load_bilstm_encoder, load_attn_encoder


# encoder, Lang = load_attn_encoder()

# conf = DIIN_conf(100,encoder)
# nov_model = DIIN(conf)


