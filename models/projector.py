import os 
import torch.nn as nn
from torch.nn import Linear,Dropout,LayerNorm

class Projector(nn.Module):
    def __init__(self,n_classes,hidden_dim=512):
        super(Projector,self).__init__()
        
        self.fc1 = Linear(1024,hidden_dim)
        self.drop = Dropout(0.25)
        self.norm = LayerNorm(hidden_dim)
        self.fc2 = Linear(hidden_dim,n_classes)
    
    def forward(self,x):
        h = self.fc1(x)
        h=self.drop(h)
        h=self.norm(h)
        h=h.relu()
        h=self.fc2(h)
        return h