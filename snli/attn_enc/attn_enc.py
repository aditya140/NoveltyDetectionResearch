import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Attn_Encoder_conf:
    embedding_dim = 300
    hidden_size = 300
    fcs = 1
    num_layers = 2
    dropout = 0.1
    opt_labels = 3
    bidirectional = True
    attn_type = "dot"
    attention_layer_param = 100
    activation = "tanh"
    freeze_embedding = False

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


class Attention(nn.Module):
    def __init__(self, conf):
        super(Attention, self).__init__()
        self.Ws = nn.Linear(
            (2 if conf.bidirectional else 1) * conf.hidden_size,
            conf.attention_layer_param,
            bias=False,
        )
        self.Wa = nn.Linear(conf.attention_layer_param, 1, bias=False)

    def forward(self, hid):
        opt = self.Ws(hid)
        opt = F.tanh(opt)
        opt = self.Wa(opt)
        opt = F.softmax(opt)
        return opt


class Attn_Encoder(nn.Module):
    def __init__(self, conf):
        super(Attn_Encoder, self).__init__()
        self.conf = conf
        self.embedding = nn.Embedding(
            num_embeddings=self.conf.vocab_size,
            embedding_dim=self.conf.embedding_dim,
            padding_idx=self.conf.padding_idx,
        )
        self.translate = nn.Linear(self.conf.embedding_dim, self.conf.hidden_size) # make (300,..) if not working
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
        self.lstm_layer = nn.LSTM(
            input_size=self.conf.hidden_size,
            hidden_size=self.conf.hidden_size,
            num_layers=self.conf.num_layers,
            bidirectional=self.conf.bidirectional,
        )
        self.attention = Attention(conf)

    def forward(self, inp):
        batch_size = inp.shape[0]
        embedded = self.embedding(inp)
        embedded = self.translate(embedded)
        embedded = self.act(embedded)
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
