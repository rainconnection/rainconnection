
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


import torch
from torch import nn

from typing import List

from embedding.feature_embedding import EmbeddingDict

class FactorizationMachine(nn.Module):
    def __init__(self, category_list : List[str], bias_yn = True):
        super(FactorizationMachine, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad = True) if bias_yn else None

        self.LogisticEmbeddingDict = EmbeddingDict(category_list, 1)

    def forward(self, x, feature_emb):
        sum_of_square = torch.sum(feature_emb, dim=1)**2
        square_of_sum = torch.sum(feature_emb**2, dim=1)

        interaction = (sum_of_square - square_of_sum) * 0.5 # [batch_size, feature_dim]

        additioned = torch.sum(self.LogisticEmbeddingDict(x), dim=1) # [batch_size, 1]
        if self.bias is not None:
            additioned += self.bias

        output = interaction + additioned # [batch_size, feature_dim]
        
        return output

