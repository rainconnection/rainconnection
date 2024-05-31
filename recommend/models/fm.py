import torch
from torch import nn

from typing import List

from ...embedding.feature_embedding import embedding_dict

class FactorizationMachine(nn.Module):
    def __init__(self, category_list : List[str], bias_yn = True):
        super(FactorizationMachine, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad = True) if bias_yn else None

        self.Logistic_by_embedding = embedding_dict(category_list, 1)

    def forward(self, X, feature_emb):
        sum_of_square = torch.sum(feature_emb, dim=1)**2
        square_of_sum = torch.sum(feature_emb**2, dim=1)

        interaction = (sum_of_square - square_of_sum) * 0.5

        additioned = torch.sum(self.Logistic_by_embedding(x), dim=1)
        if self.bias is not None:
            additioned += self.bias

        output = interaction + additioned

        return output

