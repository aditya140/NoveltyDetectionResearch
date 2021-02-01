import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ADIN_conf:
    embedding_dim = 300
    hidden_size = 400
    opt_labels = 2
    num_layers = 1

    k = 200  # attention param
    N = 2  # inference params


    num_sent = 100
    sent_len = 100
    encoder_dim = 400
    dropout = 0.3

    def __init__(self, num_sent, encoder, **kwargs):
        self.num_sent = num_sent
        self.encoder = encoder
        for k, v in kwargs.items():
            setattr(self, k, v)
    


class InferentialModule(nn.Module):
    def __init__(self, conf):
        super(InferentialModule, self).__init__()
        self.W = nn.Linear(conf.hidden_size, conf.k, bias=False)
        self.P = nn.Linear(conf.k, 1, bias=False)
        self.Wb = nn.Linear(4 * conf.hidden_size, conf.hidden_size)
        self.LayerNorm = nn.LayerNorm(conf.hidden_size)

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
            input_size=int(2 * conf.hidden_size),
            hidden_size=int(conf.hidden_size / 2),
            num_layers=conf.num_layers,
            bidirectional=True,
        )
        self.lstm_layer2 = nn.LSTM(
            input_size=int(2 * conf.hidden_size),
            hidden_size=int(conf.hidden_size / 2),
            num_layers=conf.num_layers,
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
    def __init__(self, conf):
        super(ADIN, self).__init__()
        self.conf = conf
        self.sent_len = conf.sent_len
        self.num_sent = conf.num_sent
        self.encoder = conf.encoder
        self.translate = nn.Linear(2 * self.conf.encoder_dim, self.conf.hidden_size)
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.act = nn.ReLU()
        del self.conf.encoder
        self.inference_modules = nn.ModuleList(
            [AsyncInfer(conf) for i in range(conf.N)]
        )
        self.r = nn.Linear(8 * conf.hidden_size, conf.hidden_size)
        self.v = nn.Linear(conf.hidden_size, 2)
        self.dropout = nn.Dropout(p=self.conf.dropout)



    def encode_sent(self,inp):
        batch_size,_,_ = inp.shape
        x = inp.view(-1,self.sent_len)

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
    
        embedded = self.dropout(self.translate(x_enc_t))
        embedded = self.act(embedded)
        embedded = embedded.permute(1, 0, 2)
        return embedded


    def forward(self, x0, x1, x0_char_vec=None, x1_char_vec=None):
        x0_enc = self.encode_sent(x0).permute(1,0,2)
        x1_enc = self.encode_sent(x1).permute(1,0,2)

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
