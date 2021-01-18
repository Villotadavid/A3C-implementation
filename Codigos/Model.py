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
        
        ############# image ##################
        self.Conv1=nn.Conv2d(1, 16, (3,3))
        self.Conv2=nn.Conv2d(16,32, (3,3))
        self.Fully1=nn.Linear(32*124*124,16)
        self.ReLu=nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)
        
        ############# Waypoint ###############
        self.Fully2=nn.Linear(3,9)
        self.Fully3=nn.Linear(9,16)
        
        ############# TOGETHER ###############
        self.Fully4=nn.Linear(32,16)
        self.Fully5=nn.Linear(16,9)
        
    def forward(self,img,wp):
        img=img.float()
        img=self.Conv1(img)
        img=self.ReLu(img)
        img=self.Conv2(img)
        img=img.view(-1, 124 * 124 * 32)

        img=self.ReLu(img)
        img=self.Fully1(img)
        img=self.ReLu(img) 
        
        wp=self.Fully2(wp)
        wp=self.ReLu(wp)
        wp=self.Fully3(wp)
        wp=self.ReLu(wp)
        
        x=torch.cat((img,wp),dim=1)
        
       
        x=self.Fully4(x)
        x=self.ReLu(x) 
        x=self.Fully5(x)
        x=self.ReLu(x) 
        x=self.dropout(x)
        return (x)