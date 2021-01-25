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

class res_block(nn.Module):   #nn.Module -> Base class for all neural network modules.
    def __init__(self,in_channels, out_channels,activation='relu'):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.activation=activation
        self.BN1=nn.BatchNorm2d(in_channels)
        self.BN2=nn.BatchNorm2d(out_channels)
        self.ReLu=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels, out_channels, (3, 3),stride=1,padding=1)
        self.conv2=nn.Conv2d(out_channels, out_channels, (3, 3),stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels, out_channels, (3, 3),stride=1,padding=1)
        
    def forward(self,x):
        residual=x
        out=self.BN1(x)
        out=self.ReLu(out)
        out=self.conv1(out)
        out=self.BN2(out)
        out=self.ReLu(out)
        out=self.conv2(out)
        residual=self.conv3(residual)
        out +=residual
        return out

class DQN(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.count=0
        ############# image ##################
        self.Conv1=nn.Conv2d(1, 16, (3,3))
        self.res1=res_block(16,32)
        #self.Conv2=nn.Conv2d(16,32, (3,3))
        self.Fully1=nn.Linear(32*126*126,16)
        self.ReLu=nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)
        
        ############# Waypoint ###############
        self.Fully2=nn.Linear(3,9)
        self.Fully3=nn.Linear(9,16)
        
        ############# TOGETHER ###############
        self.Fully4=nn.Linear(32,16)
        self.Fully5=nn.Linear(16,9)
        
        ############# TOGETHER ###############
        self.softmax=nn.Softmax(1)
        
    def forward(self,img,wp):
        
        img=img.float()
        img=self.Conv1(img)
        img=self.ReLu(img)
        img=self.res1(img)
        #img=self.Conv2(img)
        #img=self.ReLu(img)
        img=img.view(-1, 126 * 126 * 32)


        img=self.Fully1(img)
        img=self.ReLu(img) 
        #.register_hook(lambda grad : torch.clamp(grad, min = 0))     
        
        wp=self.Fully2(wp)
        wp=self.ReLu(wp)
        wp=self.Fully3(wp)
        wp=self.ReLu(wp)

        x=torch.cat((img,wp),dim=1)
        #x.register_hook(lambda grad: print("Gradients less than zero:", bool((grad < 0).any())))  
       
        x=self.Fully4(x)
        x=self.ReLu(x) 
        x=self.Fully5(x)
        x=self.ReLu(x)
        #print ('######'+str(self.count))
        #print (x)
        #x=self.softmax(x)

        #○x=self.dropout(x)
        self.count+=1
        return (x)