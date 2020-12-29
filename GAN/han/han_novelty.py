import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class HAN_Novelty_conf:
    encoder_dim = 600
    dropout = 0.3
    fc_hidden = 600
    freeze_encoder = False
    def __init__(self, encoder, **kwargs):
        self.encoder = encoder
        for k, v in kwargs.items():
            setattr(self, k, v)



class HAN_Novelty(nn.Module):
    def __init__(self,conf):
        super(HAN_Novelty,self).__init__()
        self.doc_encoder = conf.encoder
        self.doc_encoder.requires_grad = conf.freeze_encoder
        self.fc_in = nn.Linear(conf.encoder_dim*4,conf.fc_hidden)
        self.act = nn.ReLU()
        self.fc_out = nn.Linear(conf.fc_hidden,2)
        self.dropout = nn.Dropout(conf.dropout)
    def forward(self,x0,x1):
        x0_enc = self.doc_encoder(x0)
        x1_enc = self.doc_encoder(x1)
        cont = torch.cat(
                [
                    x0_enc,
                    x1_enc,
                    torch.abs(x0_enc - x1_enc),
                    x0_enc * x1_enc,
                ],
                dim=1,
            )
        cont = self.dropout(cont)
        cont = self.act(self.fc_in(cont))
        cont = self.fc_out(cont)
        return cont


