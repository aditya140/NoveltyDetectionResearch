import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class Transformer_config:
    embedding_dim = 512
    hidden_size = 300
    max_len = 150
    sub_enc_layer = 3
    interaction = "sum_prod"

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, conf):
        super(TransformerEncoder, self).__init__()
        self.conf = conf
        self.word_embedding = nn.Embedding(
            num_embeddings=self.conf.vocab_size,
            embedding_dim=self.conf.embedding_dim,
            padding_idx=self.conf.padding_idx,
        )

        self.pos_embedding = PositionalEncoding(self.conf.embedding_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.conf.embedding_dim, nhead=8),
            self.conf.sub_enc_layer,
        )
        self.pooler = nn.Linear(self.conf.embedding_dim, self.conf.embedding_dim)
        self.translate = nn.Linear(self.conf.embedding_dim, self.conf.embedding_dim)
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.transpose(0, 1)
        mask = self.make_src_mask(x.shape[0])
        emb = self.word_embedding(x) * math.sqrt(self.conf.embedding_dim)
        emb = self.pos_embedding(emb)
        opt = self.transformer(emb, mask)
        opt = self.pooler(opt)
        opt = self.dropout(F.relu(opt))
        return opt

    def make_src_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask.to(self.template.device)


class Transformer_snli(nn.Module):
    def __init__(self, conf):
        super(Transformer_snli, self).__init__()
        self.conf = conf
        self.encoder = TransformerEncoder(self.conf)

        if self.conf.interaction == "concat":
            final_dim = 2 * self.conf.embedding_dim
            self.interact = self.interact_concat

        elif self.conf.interaction == "sum_prod":
            final_dim = 4 * self.conf.embedding_dim
            self.interact = self.interact_sum_prod

        self.cls = nn.Linear(final_dim, 3)
        self.softmax = nn.Softmax(dim=2)

    def interact_concat(self, a, b):
        return torch.cat([a, b], dim=2)

    def interact_sum_prod(self, a, b):
        return torch.cat([a, b, a + b, a * b], dim=2)

    def forward(self, x0, x1):
        x0_emb = self.encoder(x0)[:1, :, :]
        x1_emb = self.encoder(x1)[:1, :, :]

        conc = self.interact(x0_emb,x1_emb)
        opt = self.cls(conc)

        return self.softmax(opt)
