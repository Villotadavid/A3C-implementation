# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:14:49 2020

@author: usuario
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
    )

    return block

class DQN(nn.Module):
    def __init__(self):
        self.n_channels
        self.in_dimensions
        self.out_dimensions
        self.Conv1=nn.Conv2d(in_channels, out_channels, kernel_size)
        self.Conv2=nn.Conv2d(in_channels, out_channels, kernel_size)
        self.Fully1=nn.Linear()
        self.Fully1=nn.Linear(,2)
        self.ReLu=nn.ReLU()
        
        
        
    def forward(self,x):
        out=self.Conv1(x)
        out=self.ReLu(x)
        out=self.Conv2(x)
        out=self.ReLu(x)
        out=self.Fully1(x)
        out=self.ReLu(x)
        out=self.Fully2(x)
        return (out)