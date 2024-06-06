
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


import torch
from torch import nn

from typing import List

from embedding.feature_embedding import EmbeddingDict

class GDCN(nn.Module):
    def __init__(self, category_list : List[str], vector_dim : int = 2**7):
        super(GDCN, self).__init__()

        self.embedding_vector =  EmbeddingDict(category_list, vector_dim)
        self.GCLBlock = GatedCrossLayer()




class GatedCrossLayer(nn.Module):
    def __init__(self, input_dim : int, num_gc : int, gate_function = nn.Sigmoid()):
        # input_dim : embedding_vector_dim * number of feature?
        super(GatedCrossLayer, self).__init__()

        self.num_gc = num_gc
        self.gate_function = gate_function


    def forward(self, x):
        # cross networking
        '''
        c_(l+1) = c0*(wc_l + bias_l) * gated(wg_l) + c_l
        
        wc_l : nn.linear(c_l)
        wg_l : nn.linear(c_l)
        '''