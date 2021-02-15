import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DAN_conf:
    num_sent = 100
    sent_len = 100
    encoder_dim = 400
    hidden_size = 400
    activation = 'relu'
    dropout = 0.3

    def __init__(self, num_sent, encoder, **kwargs):
        self.num_sent = num_sent
        self.encoder = encoder
        for k, v in kwargs.items():
            setattr(self, k, v)

class DAN(nn.Module):
    def __init__(self,conf,encoder):
        super(DAN,self).__init__()
        self.conf = conf
        self.sent_len = conf["max_len"]
        self.num_sent = conf["max_num_sent"]
        self.encoder = conf.encoder
        self.translate = nn.Linear(2 * self.conf["encoder_dim"], self.conf["hidden_size"])
        self.template = nn.Parameter(torch.zeros((1)), requires_grad=True)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(conf["dropout"])

        self.mlp_f = nn.Linear(self.conf["hidden_size"], self.conf["hidden_size"])
        self.mlp_g = nn.Linear(2*self.conf["hidden_size"], self.conf["hidden_size"])
        self.mlp_h = nn.Linear(2*self.conf["hidden_size"], self.conf["hidden_size"])
        self.linear = nn.Linear(self.conf["hidden_size"],2)

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


    def forward(self,x0,x1):
        x0_enc = self.encode_sent(x0).permute(1,0,2)
        x1_enc = self.encode_sent(x1).permute(1,0,2)

        f1 = self.act(self.dropout(self.mlp_f(x0_enc)))
        f2 = self.act(self.dropout(self.mlp_f(x1_enc)))

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        prob1 = F.softmax(score1.view(-1, self.num_sent)).view(-1, self.num_sent, self.num_sent)

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()

        prob2 = F.softmax(score2.view(-1, self.num_sent)).view(-1, self.num_sent, self.num_sent)

        sent1_combine = torch.cat((x0_enc, torch.bmm(prob1, x1_enc)), 2)
        sent2_combine = torch.cat((x1_enc, torch.bmm(prob2, x0_enc)), 2)

        

        g1 = self.act(self.dropout(self.mlp_g(sent1_combine)))
        g2 = self.act(self.dropout(self.mlp_g(sent2_combine)))

        sent1_output = torch.sum(g1, 1)  
        sent1_output = torch.squeeze(sent1_output, 1)
    
        sent2_output = torch.sum(g2, 1)  
        sent2_output = torch.squeeze(sent2_output, 1)


        input_combine = torch.cat((sent1_output * sent2_output, torch.abs(sent1_output - sent2_output)), 1)
        
        h = self.act(self.dropout(self.mlp_h(input_combine)))
        opt = self.linear(h)
        return opt
    

        