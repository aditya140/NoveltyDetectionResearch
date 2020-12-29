import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class HAN_conf:
    num_layers = 2
    num_sent = 100
    activation = 'relu'
    dropout = 0.3
    encoder_dim = 800
    bidirectional = True
    hidden_size = 300
    attention_layer_param = 50
    fcs = 1
    opt_labels = 10

    def __init__(self, num_sent, encoder, **kwargs):
        self.num_sent = num_sent
        self.encoder = encoder
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


class HAN(nn.Module):
    def __init__(self,conf):
        super(HAN,self).__init__()
        self.conf = conf
        self.num_sent = conf.num_sent
        self.encoder = conf.encoder
        del self.conf.encoder
        self.translate = nn.Linear(self.conf.encoder_dim, self.conf.hidden_size) # make (300,..) if not working
        if self.conf.activation.lower() == "relu".lower():
            self.act = nn.ReLU()
        elif self.conf.activation.lower() == "tanh".lower():
            self.act = nn.Tanh()
        elif self.conf.activation.lower() == "leakyrelu".lower():
            self.act = nn.LeakyReLU()
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.lstm_layer = nn.LSTM(
            input_size=self.conf.hidden_size,
            hidden_size=self.conf.hidden_size,
            num_layers=self.conf.num_layers,
            bidirectional=self.conf.bidirectional,
        )
        self.attention = Attention(conf)

    def forward(self,inp):
        batch_size,_,_ = inp.shape
        x = inp.view(-1,self.num_sent)

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

        embedded = self.translate(x_enc_t)
        embedded = self.act(embedded)
        embedded = embedded.permute(1, 0, 2)
        all_, (hid, cell) = self.lstm_layer(embedded)

        attn = self.attention(all_)

        cont = torch.bmm(all_.permute(1, 2, 0), attn.permute(1, 0, 2)).permute(2, 0, 1)
        return cont.squeeze(0)

class HAN_classifier(nn.Module):
    def __init__(self,conf):
        super(HAN_classifier,self).__init__()
        self.han = HAN(conf)

        if conf.activation.lower() == "relu".lower():
            self.act = nn.ReLU()
        elif conf.activation.lower() == "tanh".lower():
            self.act = nn.Tanh()
        elif conf.activation.lower() == "leakyrelu".lower():
            self.act = nn.LeakyReLU()

        self.fc_in = nn.Linear(
            (2 if conf.bidirectional else 1)  * conf.hidden_size,
            conf.hidden_size,
        )

        self.fcs = nn.ModuleList(
            [
                nn.Linear(conf.hidden_size, conf.hidden_size)
                for i in range(conf.fcs)
            ]
        )
        self.fc_out = nn.Linear(conf.hidden_size, conf.opt_labels)
        self.dropout = nn.Dropout(conf.dropout)

                 
    def forward(self,inp):
        cont=self.han(inp)
        opt = self.fc_in(cont)
        opt = self.dropout(opt)
        for fc in self.fcs:
            opt = fc(opt)
            opt = self.dropout(opt)
            opt = self.act(opt)
        opt = self.fc_out(opt)
        return opt




class HAN_regressor(nn.Module):
    def __init__(self,conf):
        super(HAN_regressor,self).__init__()
        self.han = HAN(conf)

        if conf.activation.lower() == "relu".lower():
            self.act = nn.ReLU()
        elif conf.activation.lower() == "tanh".lower():
            self.act = nn.Tanh()
        elif conf.activation.lower() == "leakyrelu".lower():
            self.act = nn.LeakyReLU()

        self.fc_in = nn.Linear(
            (2 if conf.bidirectional else 1)  * conf.hidden_size,
            conf.hidden_size,
        )

        self.fcs = nn.ModuleList(
            [
                nn.Linear(conf.hidden_size, conf.hidden_size)
                for i in range(conf.fcs)
            ]
        )
        self.fc_out = nn.Linear(conf.hidden_size,1)
        self.dropout = nn.Dropout(conf.dropout)

                 
    def forward(self,inp):
        cont=self.han(inp)
        opt = self.fc_in(cont)
        opt = self.dropout(opt)
        for fc in self.fcs:
            opt = fc(opt)
            opt = self.dropout(opt)
            opt = self.act(opt)
        opt = self.fc_out(opt)
        return opt


        


# model_conf = HAN_conf(100,encoder)
# model = HAN_classifier(model_conf)