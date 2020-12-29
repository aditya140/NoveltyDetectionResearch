import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.init import kaiming_uniform_
import math


class Struc_Attn_Encoder_conf:
    embedding_dim = 300
    hidden_size = 300
    fcs = 1
    r = 30
    num_layers = 2
    dropout = 0.1
    opt_labels = 3
    bidirectional = True
    attn_type = "dot"
    attention_layer_param = 100
    activation = "tanh"
    freeze_embedding = False
    gated_embedding_dim = 100
    gated = True
    pool_strategy = 'max' # max,avg
    penalty =True
    C = 0

    def __init__(self, lang, embedding_matrix=None, **kwargs):
        self.embedding_matrix = None
        if lang.tokenizer_ == "BERT":
            self.vocab_size = lang.vocab_size
            self.padding_idx = lang.bert_tokenizer.vocab["[PAD]"]
        else:
            self.embedding_matrix = embedding_matrix
            self.vocab_size = lang.vocab_size_final()
            self.padding_idx = lang.word2idx[lang.config.pad]
        for k, v in kwargs.items():
            setattr(self, k, v)


class Struc_Attention(nn.Module):
    def __init__(self, conf):
        super(Struc_Attention, self).__init__()
        self.Ws = nn.Linear(
            (2 if conf.bidirectional else 1) * conf.hidden_size,
            conf.attention_layer_param,
            bias=False,
        )
        self.Wa = nn.Linear(conf.attention_layer_param, conf.r, bias=False)

    def forward(self, hid):
        opt = self.Ws(hid)
        opt = F.tanh(opt)
        opt = self.Wa(opt)
        opt = F.softmax(opt)
        return opt


class Struc_Attn_Encoder(nn.Module):
    def __init__(self, conf):
        super(Struc_Attn_Encoder, self).__init__()
        self.conf = conf
        self.embedding = nn.Embedding(
            num_embeddings=self.conf.vocab_size,
            embedding_dim=self.conf.embedding_dim,
            padding_idx=self.conf.padding_idx,
        )
        self.translate = nn.Linear(self.conf.embedding_dim, self.conf.hidden_size) # make (300,..) if not working
        self.init_activation()
        if isinstance(self.conf.embedding_matrix, np.ndarray):
            self.embedding.from_pretrained(
                torch.tensor(self.conf.embedding_matrix),
                freeze=self.conf.freeze_embedding,
            )
        self.lstm_layer = nn.LSTM(
            input_size=self.conf.hidden_size,
            hidden_size=self.conf.hidden_size,
            num_layers=self.conf.num_layers,
            bidirectional=self.conf.bidirectional,
        )
        self.attention = Struc_Attention(conf)
    
    def init_activation(self):
        if self.conf.activation.lower() == "relu".lower():
            self.act = nn.ReLU()
        elif self.conf.activation.lower() == "tanh".lower():
            self.act = nn.Tanh()
        elif self.conf.activation.lower() == "leakyrelu".lower():
            self.act = nn.LeakyReLU()

    def forward(self, inp):
        batch_size = inp.shape[0]
        embedded = self.embedding(inp)
        embedded = self.translate(embedded)
        embedded = self.act(embedded)
        embedded = embedded.permute(1, 0, 2)
        all_, (hid, cell) = self.lstm_layer(embedded)
        attn = self.attention(all_)
        cont = torch.bmm(all_.permute(1, 2, 0), attn.permute(1, 0, 2)).permute(2, 0, 1)
        return cont,attn


class Struc_Attn_encoder_snli(nn.Module):
    def __init__(self, conf):
        super(Struc_Attn_encoder_snli, self).__init__()
        self.conf = conf
        self.encoder = Struc_Attn_Encoder(conf)
        self.gated = self.conf.gated
        self.penalty = self.conf.penalty
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.pool_strategy = self.conf.pool_strategy
        # Gated parameters
        if self.gated :
            self.wt_p = torch.nn.Parameter(
            torch.rand(
                    (self.conf.r,
                    (2 if conf.bidirectional else 1) * self.conf.hidden_size,
                    self.conf.gated_embedding_dim
                    )
                ))
            self.wt_h = torch.nn.Parameter(
                torch.rand(
                        (self.conf.r,
                        (2 if conf.bidirectional else 1) * self.conf.hidden_size,
                        self.conf.gated_embedding_dim
                        )
                    ))
            self.init_gated_encoder()
            self.fc_in = nn.Linear(
                self.conf.gated_embedding_dim * self.conf.r,
                self.conf.hidden_size,
            )
            self.fcs = nn.ModuleList(
                [
                    nn.Linear(self.conf.hidden_size, self.conf.hidden_size)
                    for i in range(self.conf.fcs)
                ]
            )
            self.fc_out = nn.Linear(self.conf.hidden_size, self.conf.opt_labels)

        # Non Gated Version (max pool avg pool)
        else:
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


        self.init_activation()
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=self.conf.dropout)
        

    def init_gated_encoder(self):
        kaiming_uniform_(self.wt_p)
        kaiming_uniform_(self.wt_h)

    def init_activation(self):
        if self.conf.activation.lower() == "relu".lower():
            self.act = nn.ReLU()
        elif self.conf.activation.lower() == "tanh".lower():
            self.act = nn.Tanh()
        elif self.conf.activation.lower() == "leakyrelu".lower():
            self.act = nn.LeakyReLU()

    def penalty_l2(self,att):
        att = att.permute(1,0,2)
        penalty = (torch.norm(torch.bmm(att,att.transpose(1,2)) - torch.eye(att.size(1)).to(self.template.device),p='fro')/att.size(0))**2
        return penalty

    def forward(self, x0, x1):
        x0_enc,x0_attn = self.encoder(x0.long())
        x0_enc = self.dropout(x0_enc)
        x1_enc,x1_attn = self.encoder(x1.long())
        x1_enc = self.dropout(x1_enc)

        if self.gated:
            F0 = x0_enc @ self.wt_p
            F1 = x1_enc @ self.wt_h
            
            Fr = F0*F1

            Fr = Fr.permute(1,0,2).flatten(start_dim=1)
        else:
            if self.pool_strategy == 'avg':
                F0 = x0_enc.mean(0,keepdim=True)
                F1 = x1_enc.mean(0,keepdim=True)
                Fr = torch.cat(
                    [F0, F1, torch.abs(F0 - F1), F0 * F1], dim=2
                )
            elif self.pool_strategy == 'max':
                F0 = x0_enc.max(0,keepdim=True)
                F0 = F0.values
                F1 = x1_enc.max(0,keepdim=True)
                F1 = F1.values
                Fr = torch.cat(
                    [F0, F1, torch.abs(F0 - F1), F0 * F1], dim=2
                )
        opt = self.fc_in(Fr)
        opt = self.dropout(opt)
        for fc in self.fcs:
            opt = fc(opt)
            opt = self.dropout(opt)
            opt = self.act(opt)
        opt = self.fc_out(opt)
        if self.penalty:
            return opt,x0_attn,x1_attn
        return opt
