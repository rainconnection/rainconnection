import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import time

from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn
from torch import tensor
import torch.nn.functional as F


class NLinear(nn.Module):
    def __init__(self, train_size, forecast_size):
        super(NLinear, self).__init__()
        
        self.train_size = train_size
        self.forecast_size = forecast_size
        self.layers = nn.Linear(self.train_size, self.forecast_size)
        
        nn.init.xavier_normal_(self.layers.weight.data)
        
    def forward(self, x):
        last = x[:,-1].clone().unsqueeze(1)
        x -= last
        x = self.layers(x)
        x += last
        return x
    
class moving_avg(nn.Module):
    def __init__(self, kernel_size):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x)
        x = x
        return x
    
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    def __init__(self, train_size, forecast_size, kernal_size):
        super(DLinear, self).__init__()
        
        self.decomp = series_decomp(kernal_size)
        self.NLinear = NLinear(train_size, forecast_size)
        
    def forward(x):
        
        res, trend = self.decomp(x)
        
        res = self.Nlinear(res)
        trend = self.Nlinear(trend)
        
        x = trend + res
        
        return x
    
