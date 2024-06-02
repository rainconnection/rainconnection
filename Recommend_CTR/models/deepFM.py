
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import torch
from torch import nn
from embedding.feature_embedding import FeatureEmbeddingModel
from fm import FactorizationMachine

from typing import List

class DeepFM(nn.Module):
    def __init__(self, category_list : List[str],
                 hidden_dims : List[int] = [16, 16, 16],
                 hidden_act = nn.ReLU(),
                 bias_yn = True,
                 dropout = None):
        super(DeepFM, self).__init__()

        self.fm_block = FactorizationMachine(category_list, bias_yn)
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.hidden_layers.append(hidden_act)
            if dropout is not None:
                self.hidden_layers.append(dropout)

        self.hidden_layers = nn.Sequential(self.hidden_layers)

    def forward(self, x, feature_emb):
        fm_output = self.fm_block(x, feature_emb) # [batch_size, feature_dim]
        higher_output = self.hidden_layers(feature_emb) # [batch_size, feature_dim, last_hidden_dim]

        output = fm_output + higher_output.sum(dim=1)

        return output



