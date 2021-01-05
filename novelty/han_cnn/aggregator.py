import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Accumulator(nn.Module):
    def __init__(self, conf):
        super(Accumulator, self).__init__()
        self.conf = conf
        self.encoder = self.conf.encoder.encoder
        self.encoder.requires_grad = conf.freeze_encoder
        self.num_sent = conf.num_sent
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)

    def forward(self, src, trg):
        batch_size = src.shape[0]

        x = src.view(-1, self.num_sent)
        y = trg.view(-1, self.num_sent)

        x_padded_idx = x.sum(dim=1) != 0
        y_padded_idx = y.sum(dim=1) != 0

        x_enc = []

        for sub_batch in x[x_padded_idx].split(64):
            x_enc.append(self.encoder(sub_batch)[0])
        x_enc = torch.cat(x_enc, dim=0)
        y_enc = []

        for sub_batch in y[y_padded_idx].split(64):
            y_enc.append(self.encoder(sub_batch)[0])
        y_enc = torch.cat(y_enc, dim=0)

        x_enc_t = torch.zeros((batch_size * self.num_sent, x_enc.size(1))).to(
            self.template.device
        )
        x_enc_t[x_padded_idx] = x_enc

        y_enc_t = torch.zeros((batch_size * self.num_sent, y_enc.size(1))).to(
            self.template.device
        )
        y_enc_t[y_padded_idx] = y_enc

        x_enc_t = x_enc_t.view(batch_size, self.num_sent, -1)
        y_enc_t = y_enc_t.view(batch_size, self.num_sent, -1)
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
        if self.conf.expand_features:
            rdv = torch.cat(
                [
                    x_enc_t,
                    matched_y,
                    torch.abs(x_enc_t - matched_y),
                    x_enc_t * matched_y,
                ],
                dim=2,
            )
        else:
            rdv = torch.cat([x_enc_t, matched_y], dim=2)
        return rdv
