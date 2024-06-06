
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import torch
from torch import nn
from embedding.feature_embedding import FeatureEmbeddingModel
from .fm import FactorizationMachine

from typing import List

class DeepFM(nn.Module):
    def __init__(self, category_list : List[str],
                 hidden_dims : List[int] = [16, 16, 16],
                 hidden_act = nn.ReLU(),
                 bias_yn = True,
                 dropout_ratio = None,
                 pooling_type = 'sum',
                 output_act = None):
        super(DeepFM, self).__init__()

        # Factorization Machine for fm layer
        self.fm_layer = FactorizationMachine(category_list, bias_yn)

        # sequential layer for higher interaction
        layer_list = nn.ModuleList()
        self.pooling_type = pooling_type
        self.output_act = output_act

        for i in range(len(hidden_dims)-1):
            layer_list.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layer_list.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layer_list.append(hidden_act)
            if dropout_ratio is not None:
                layer_list.append(nn.Dropout(dropout_ratio))

        self.hidden_layers = nn.Sequential(*layer_list)

    def forward(self, x, feature_emb):
        fm_output = self.fm_layer(x, feature_emb) # [batch_size, feature_dim]
        higher_output = self.hidden_layers(feature_emb) # [batch_size, feature_dim, last_hidden_dim]

        if self.pooling_type == 'sum':
            output = fm_output + higher_output.sum(dim=1) # [batch_size, feature_dim]
        elif self.pooling_type == 'flatten':
            output = fm_output + higher_output.flatten(start_dim=1) # [batch_size, feature_dim * last_hidden_dim] = [batch_size, hidden_dims[-1]]

        if self.output_act is not None:
            output = self.output_act(output)
        
        return output
    
    def fit(self, x, feature_emb):
        pass

    def _do_batch(self):
        pass