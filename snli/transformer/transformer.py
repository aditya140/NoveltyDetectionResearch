import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class transformer_config:
    embedding_dim = 300
    hidden_size = 300


class TransformerEncoder(nn.Module):
    def __init__(self, conf):
        super(TransformerEncoder, self).__init__()
        self.word_embedding = nn.Embedding(
            num_embeddings=self.conf.vocab_size,
            embedding_dim=self.conf.embedding_dim,
            padding_idx=self.conf.padding_idx,
        )
        self.pos_embeddding = nn.Embedding(
            num_embeddings=self.conf.max_len, embedding_dim=self.conf.embedding_dim
        )
        self.transformer = nn.TransformerEncoder(
            self.conf.encoder_layers, self.conf.sub_enc_layer, norm=False
        )
        
    def forward(self,)