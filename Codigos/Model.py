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
        super().__init__()
        self.Conv1=nn.Conv2d(1, 16, (3,3))
        self.Conv2=nn.Conv2d(16,32, (3,3))
        self.Fully1=nn.Linear(32*124*124,16)
        self.Fully2=nn.Linear(16,2)
        self.ReLu=nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)
        
        
        
    def forward(self,x):
        print ('hola')
        x=self.Conv1(x)
        print('1',x.size())
        x=self.ReLu(x)
        print('2',x.size())
        x=self.Conv2(x)
        print('3',x.size())
        x = x.view(-1, 124 * 124 * 32)
        x=self.ReLu(x)
        print('4',x.size())
        x=self.Fully1(x)
        print('5',x.size())
        x=self.ReLu(x)
        print('6',x.size())
        x=self.Fully2(x)
        print('7',x.size())
        x=self.dropout(x)
        return (x)