import torch
from torch import nn
from torch import tensor


def get_loss(infered, label, loss_f, opt = None):
    loss_func = loss_f
        
    loss = loss_func(infered, label)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), len(label) 

class custom_loss(nn.Module):
    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func
    def forward(self, y_pred, y):
        loss = self.loss_func(y_pred, y)
        loss = loss.mean()
        
        return loss

    def weighted_l1loss(y_pred, y):
        diff = y - y_pred
        loss = torch.where(diff >= 0,
                           (y - y_pred).abs(),
                           (y - y_pred).abs()*2)
        return loss

    def smape_loss(y_pred, y):
        loss = 2*(y - y_pred).abs() / (y.abs() + y_pred.abs())
        return loss