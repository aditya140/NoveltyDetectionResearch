import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ADIN_encoder_conf:
    embedding_dim = 300
    hidden_size = 300
    dropout = 0.1
    opt_labels = 3
    attention_layer_param = 100
    char_embedding_size = 50
    num_layers = 1

    k = 100  # attention param
    N = 2  # inference params

    def __init__(self, lang, embedding_matrix=None, **kwargs):
        self.embedding_matrix = None
        self.char_emb = lang.char_emb
        self.char_vocab_size = lang.char_vocab_size
        self.char_word_len = lang.char_emb_max_len

        if lang.tokenizer_ == "BERT":
            self.vocab_size = lang.vocab_size
            self.padding_idx = lang.bert_tokenizer.vocab["[PAD]"]
        else:
            self.embedding_matrix = embedding_matrix
            self.vocab_size = lang.vocab_size_final()
            self.padding_idx = lang.word2idx[lang.config.pad]
        for k, v in kwargs.items():
            setattr(self, k, v)


class SentenceEncoder(nn.Module):
    def __init__(self, conf):
        super(SentenceEncoder, self).__init__()
        self.conf = conf
        self.embedding = nn.Embedding(
            num_embeddings=self.conf.vocab_size,
            embedding_dim=self.conf.embedding_dim,
            padding_idx=self.conf.padding_idx,
        )

        if self.conf.char_emb:
            self.char_embedding = nn.Embedding(
                num_embeddings=self.conf.char_vocab_size,
                embedding_dim=self.conf.char_embedding_size,
                padding_idx=0,
            )
            self.char_cnn = nn.Conv2d(
                self.conf.char_word_len,
                self.conf.char_embedding_size,
                (1, 6),
                stride=(1, 1),
                padding=0,
                bias=True,
            )

        self.translate = nn.Linear(
            self.conf.embedding_dim
            + (self.conf.char_embedding_size if self.conf.char_emb else 0),
            self.conf.hidden_size,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=self.conf.dropout)

        if isinstance(self.conf.embedding_matrix, np.ndarray):
            self.embedding.from_pretrained(
                torch.tensor(self.conf.embedding_matrix),
                freeze=self.conf.freeze_embedding,
            )

    def char_embedding_forward(self, x):
        batch_size, seq_len, char_emb_size = x.shape
        x = x.view(-1, char_emb_size)
        x = self.char_embedding(x)
        x = x.view(batch_size, -1, seq_len, char_emb_size)
        x = x.permute(0, 3, 2, 1)
        x = self.char_cnn(x)
        x = torch.max(F.relu(x), 3)[0]
        return x.view(-1, seq_len, self.conf.char_embedding_size)

    def forward(self, inp, char_vec=None):
        batch_size = inp.shape[0]
        embedded = self.embedding(inp)
        if char_vec != None:
            char_emb = self.char_embedding_forward(char_vec)
            embedded = torch.cat([embedded, char_emb], dim=2)
        embedded = self.dropout(embedded)
        embedded = self.translate(embedded)
        embedded = self.dropout(self.act(embedded))
        # embedded = embedded.permute(1, 0, 2)
        return embedded


class InferentialModule(nn.Module):
    def __init__(self, conf):
        super(InferentialModule, self).__init__()
        self.W = nn.Linear(conf.embedding_dim, conf.k, bias=False)
        self.P = nn.Linear(conf.k, 1, bias=False)
        self.Wb = nn.Linear(4 * conf.embedding_dim, conf.embedding_dim)
        self.LayerNorm = nn.LayerNorm(conf.embedding_dim)

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
        self.encoder = SentenceEncoder(conf)
        self.inference_modules = nn.ModuleList(
            [AsyncInfer(conf) for i in range(conf.N)]
        )
        self.r = nn.Linear(8 * conf.hidden_size, conf.hidden_size)
        self.v = nn.Linear(conf.hidden_size, 3)
        self.dropout = nn.Dropout(p=self.conf.dropout)

    def forward(self, x0, x1, x0_char_vec=None, x1_char_vec=None):
        x0_enc = self.encoder(x0.long(), char_vec=x0_char_vec)
        x0_enc = self.dropout(x0_enc)
        x1_enc = self.encoder(x1.long(), char_vec=x1_char_vec)
        x1_enc = self.dropout(x1_enc)

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
        y = F.softmax(self.v(v))
        return y
