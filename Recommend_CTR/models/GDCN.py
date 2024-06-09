
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


import torch
from torch import nn

from typing import List

from embedding.feature_embedding import EmbeddingDict

class GDCN(nn.Module): # 기본 구조. Stack, parallel로 세분화 필요. 2개로?
    def __init__(self, category_list : List[str], vector_dim : int = 2**7):
        super(GDCN, self).__init__()

        self.embedding_vector =  EmbeddingDict(category_list, vector_dim)
        self.GCLBlock = GatedCrossLayer()




class GatedCrossLayerBlock(nn.Module):
    def __init__(self, input_dim : int, num_layers : int, gate_function = nn.Sigmoid()):
        # input_dim : embedding_vector_dim * number of feature?
        super(GatedCrossLayerBlock, self).__init__()
        input_dim = input_dim
        self.num_layers = num_layers
        self.gate_function = gate_function
        
        self.wc = nn.ModuleList() # weight of cross layer
        self.wg = nn.ModuleList() # weight of gate layer
        self.bias = nn.ModuleList() # bias
        for _ in range(self.num_layers):
            self.wc.append(nn.Linear(input_dim, input_dim))
            self.wg.append(nn.Linear(input_dim, input_dim))
            self.bias.append(
                nn.Parameter(torch.zeroes((input_dim,)))
                )

    def forward(self, x):
        # cross networking
        '''
        c_(l+1) = c0*(wc_l + bias_l) * gated(wg_l) + c_l
        
        wc_l : nn.linear(c_l)
        wg_l : nn.linear(c_l)
        '''
        x0 = x
        for i in range(self.num_layers):
            xc = self.wc[i](x)
            xg = self.gate_function(self.wg[i](x))

            x = x0*(xc+self.bias[i])*xg + x

        return x # [input_dim]