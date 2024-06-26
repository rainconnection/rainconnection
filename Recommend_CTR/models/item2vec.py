
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from typing import List

from embedding.feature_embedding import FeatureEmbeddingModel

from time import time


class item2vec(FeatureEmbeddingModel):
    
    def fit(self, train_data, epochs, optimizer):
        self.optimizer = optimizer

        train_loader = TensorDataset(self.embedding_dict(train_data[0]),
                                     self.embedding_dict(train_data[1]))
        train_loader = DataLoader(train_loader)

        #do train
        now = time.time()
        for epoch in range(epochs):
            losses, nums = self._batch_process(train_loader)
            train_loss = np.sum(np.multiply(losses, nums))/np.sum(nums)
            print('train loss : ', train_loss, ', training_time : ', time.time() - now)

