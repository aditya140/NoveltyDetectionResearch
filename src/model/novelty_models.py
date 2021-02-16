import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.model.nli_models import *


"""
Decomposable Attention Network
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
        prob1 = F.softmax(score1.view(-1, self.num_sent)).view(
            -1, self.num_sent, self.num_sent
        )

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()

        prob2 = F.softmax(score2.view(-1, self.num_sent)).view(
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
Asynchronous Deep Interactive Network
"""


class InferentialModule(nn.Module):
    def __init__(self, conf):
        super(InferentialModule, self).__init__()
        self.W = nn.Linear(conf["hidden_size"], conf["k"], bias=False)
        self.P = nn.Linear(conf["k"], 1, bias=False)
        self.Wb = nn.Linear(4 * conf["hidden_size"], conf["hidden_size"])
        self.LayerNorm = nn.LayerNorm(conf["hidden_size"])

    def forward(self, ha, hb):
        e = F.softmax(self.P(F.tanh(self.W(ha * hb))))
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
Hierarchical Attention Network
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
