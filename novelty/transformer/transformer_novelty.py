import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Transformer_conf:
    num_sent = 100
    sent_len = 100
    encoder_dim = 400
    hidden_size = 768
    activation = "relu"
    dropout = 0.3
    transformer_max_len = num_sent * 2 + 1
    n_heads = 6
    sub_enc_layer = 1

    def __init__(self, num_sent, encoder, **kwargs):
        self.num_sent = num_sent
        self.encoder = encoder
        for k, v in kwargs.items():
            setattr(self, k, v)


class Transformer_novelty(nn.Module):
    def __init__(self, conf):
        super(Transformer_novelty, self).__init__()
        self.conf = conf
        self.sent_len = conf.sent_len
        self.num_sent = conf.num_sent
        self.encoder = conf.encoder
        del self.conf.encoder
        self.translate = nn.Linear(2 * self.conf.encoder_dim, self.conf.hidden_size)
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        if self.conf.activation.lower() == "relu".lower():
            self.act = nn.ReLU()
        elif self.conf.activation.lower() == "tanh".lower():
            self.act = nn.Tanh()
        elif self.conf.activation.lower() == "leakyrelu".lower():
            self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(conf.dropout)

        self.pos_embedding = nn.Embedding(
            num_embeddings=self.conf.transformer_max_len,
            embedding_dim=self.conf.hidden_size,
        )
        self.register_buffer(
            "position_ids", torch.arange(self.conf.transformer_max_len).expand((1, -1))
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.conf.hidden_size, nhead=self.conf.n_heads
            ),
            self.conf.sub_enc_layer,
        )

        self.LayerNorm = nn.LayerNorm(self.conf.hidden_size)
        self.pooler = nn.Linear(self.conf.hidden_size, self.conf.hidden_size)

        self.translate_trans = nn.Linear(self.conf.hidden_size, self.conf.hidden_size)
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.dropout = nn.Dropout(p=0.3)
        self.cls = nn.Linear(self.conf.hidden_size, 2)

    def encode_sent(self, inp):
        batch_size, _, _ = inp.shape
        x = inp.view(-1, self.sent_len)

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

    def forward(self, x0, x1):
        batch_size, _, _ = x0.shape
        x0_enc = self.encode_sent(x0).permute(1, 0, 2)
        sep_token = torch.zeros((batch_size, 1, self.conf.hidden_size)).to(
            self.template.device
        )
        x1_enc = self.encode_sent(x1).permute(1, 0, 2)
        emb = torch.cat([x0_enc, sep_token, x1_enc], dim=1)
        emb = emb.permute(1, 0, 2)
        # print(emb.shape)

        position_ids = self.position_ids.expand(batch_size, -1).transpose(0, 1)
        # print(position_ids.shape)

        pos_embedding = self.pos_embedding(position_ids)
        # print(pos_embedding.shape)
        emb = emb + pos_embedding

        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)

        opt = self.transformer(emb)[:1, :, :]
        opt = self.pooler(opt)
        opt = self.dropout(F.tanh(opt))
        opt = self.translate_trans(opt)
        opt = self.cls(opt)
        opt = opt.permute(1, 0, 2)
        return opt
