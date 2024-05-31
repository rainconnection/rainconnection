import torch
import torch.nn as nn
import torch.nn.functional as F
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.hidden1 = nn.Linear(30,64)
        self.hidden2 = nn.Linear(64,128)

        self.bn2 = nn.BatchNorm1d(64)
        self.Encoder = nn.Sequential(
            self.hidden1,
            nn.BatchNorm1d(64),
            nn.SELU(),
            self.hidden2,
            nn.BatchNorm1d(128),
            nn.SELU(),
            #self.hidden3,
            #nn.BatchNorm1d(256),
            #nn.SELU(),
        )
        
    def forward(self, x):
        x = self.Encoder(x)
        x = torch.matmul(x, self.hidden2.weight)
        x = self.bn2(x)
        x = F.selu(x)
        x = torch.matmul(x, self.hidden1.weight)
        return x