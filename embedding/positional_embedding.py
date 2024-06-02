import torch
from torch import nn, tensor

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float() # [max_len, d_model]
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1) # [max_len, 1]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp() # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term) # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term) # [max_len, d_model/2]

        pe = pe.unsqueeze(0) # [1,max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]