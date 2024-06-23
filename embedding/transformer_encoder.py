from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from rainconnection.embedding.positional_encoder import PositionalEncoder

class TransformerEncoder(nn.Module):
    def __init__(self, d_model : int, nhead : int, multiple_FF : int = 4,
                 dropout = 0.1, batch_first = True,
                 adapter_size = 2**6,
                 n_encoder: int = 1
                ):
        super(self, TransformerEncoder).__init__()
        

        TELayer = nn.ModuleList()
        for _ in range(n_encoder):
            TELayer.append(TransformerEncoderLayer(d_model, nhead,
                                                multiple_FF, dropout,
                                                batch_first,
                                                adapter_size))
        self.TELayer = nn.Sequential(TELayer)
        self.PE = PositionalEncoder(d_model)

    def forward(self, x):
        x = self.PE(x)
        x = self.TELayer(x)
        return x
        

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model : int, nhead : int, multiple_FF : int = 4,
                 dropout = 0.1, batch_first = True,
                 adapter_size = 2**6
                ):
        super(self, TransformerEncoderLayer).__init__()

        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout = dropout,
                                                    batch_first = batch_first)
        
        ### multihead attention layer
        self.layer1 = nn.Sequential(
            self.self_attention,
            nn.Dropout(dropout),
            Adapter(input_dim = d_model, hidden_dim = adapter_size, output_dim = d_model)
        )
        self.norm1 = nn.LayerNorm(d_model, eps = 1e-5)
        
        ### FeedForward Layer
        self.layer2 = nn.Sequential(
            nn.Linear(d_model, d_model*multiple_FF),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*multiple_FF, d_model),
            nn.Dropout(dropout),
            Adapter(input_dim = d_model, hidden_dim = adapter_size, output_dim = d_model)
        )
        self.norm2 = nn.LayerNorm(d_model, eps = 1e-5)
        
    def forward(self, x):
        _x = self.layer1(x)
        x = self.norm1(x + _x)

        _x = self.layer2(x)
        x = self.norm2(x + _x)

        return x
    


class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(self, Adapter).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.GELU()
        nn.init.kaiming_normal(self.encoder)
        nn.init.kaiming_normal(self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        return x
    

