
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