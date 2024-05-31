import torch
from torch import nn, tensor

from typing import List

from time import time

class FeatureEmbeddingModel(nn.Module):
    def __init__(self, items : List[str] = ['test_item_1', 'test_item_2'],
                vector_dim = 2**7,
                activation_function = nn.Sigmoid()):
        super(FeatureEmbeddingModel, self).__init__()

        self.embedding_dict = EmbeddingDict(items, vector_dim)
        self.activation_function = activation_function

        
    def forward(self, pair_items):
        target = pair_items[0]
        context = pair_items[1]

        output = torch.sum(target * context, dim=1)
        output = self.activation_function(output)

        return output
    
    def fit(self, train_loader, epochs, optimizer):
        target = self.embedding_dict(pair_items[0])
        context = self.embedding_dict(pair_items[1])
        ## do train
        pass

        
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
    
    def _get_loss(self, infered, label):
        loss_func = nn.MSELoss()

        loss = loss_func(infered, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), len(infered)
        




class EmbeddingDict(nn.Module):
    def __init__(self, items : List[str], vector_dim = 2**7):
        super(EmbeddingDict, self).__init__()
        self.vector_dim = vector_dim
        self.items = set()

        self.item2idx = {}
        self.idx2item = {}
        
        self.vectors = nn.Embedding(1, self.vector_dim, padding_idx = 0)
        
        self.add_item(items)


    def add_item(self, add_items : List[str]):
        new_items = set()

        start_ids = len(self.items)
        for item in add_items:
            if item not in self.items:
                self.items.add(item)
                new_items.add(item)


        new_vectors = torch.randn(len(new_items), self.vector_dim)
        nn.init.xavier_normal_(new_vectors)

        self.vectors = nn.Embedding.from_pretrained(torch.cat([self.vectors.weight, new_vectors]), padding_idx = 0)
        self.vectors.weight.requires_grad = True

        for i, item in zip(range(start_ids, len(self.items)), new_items):
            ids = i+1
            self.item2idx[item] = ids
            self.idx2item[ids] = item

        
    def forward(self, call_items : List[str]):
        calls = []
        for item in call_items:
            calls.append(self.item2idx[item])

        return self.vectors(tensor(calls))