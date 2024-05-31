import torch
from torch import nn, tensor

import numpy as np
from typing import List

from time import time


class item2vec(nn.Module):
    def __init__(self, vector_dim : int = 2**7, activation_function = nn.Sigmoid()):
        super(item2vec, self).__init__()
        
        self.vector_dim = vector_dim
        self.items = set()
        self.embedding_vector = nn.Embedding(1, self.vector_dim, padding_idx = 0)
        self.activation_function = activation_function


    def add_item(self, add_items : List[str]):
        new_items = []

        start_ids = len(self.items)
        for item in add_items:
            if item not in self.items:
                self.items.add(item)
                new_items.append(item)


        new_embedding_vector = torch.randn(len(new_items), self.vector_dim)
        nn.init.xavier_normal_(new_embedding_vector)

        self.embedding_vector = nn.Embedding.from_prtrained(torch.cat([self.embedding_vecotr.weight, new_embedding_vector]), padding_idx = 0)
        self.embedding_vector.weight.requires_grad = True

        for i, item in zip(range(start_ids, len(self.items)), new_items):
            ids = i+1
            self.item2idx[item] = ids
            self.idx2item[ids] = item

        
    def forward(self, batch_data):
        x = batch_data[0]
        y = batch_data[1]

        target = self.embedding_vector(x)
        context = self.embedding_vector(y)

        output = torch.sum(target * context, dim=1)
        output = self.activation_function(output)

        return output
    
    def fit(self, train_loader, epochs, optimizer):
        self.optimizer = optimizer
        #do train
        now = time.time()
        for epoch in range(epochs):
            losses, nums = self._batch_process(train_loader)
            train_loss = np.sum(np.multiply(losses, nums))/np.sum(nums)
            print('train loss : ', train_loss, ', training_time : ', time.time() - now)

        
    def _batch_process(self, data):
        losses = []
        nums = []
        for b in data:
            _score = self.forward(b)

            loss, num = self._get_loss(infered = _score,
                                       label = tensor([1.0 for _ in range(_score.shape[0])])
                                       )
            
            losses.append(loss)
            nums.append(num)
        
        return losses, nums
    
    def get_loss(self, infered, label):
        loss_func = nn.MSELoss()

        loss = loss_func(infered, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), len(infered)
