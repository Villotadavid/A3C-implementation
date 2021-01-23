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

class res_block(nn.Module):   #nn.Module -> Base class for all neural network modules.
    def __init__(self,in_channels, out_channels,activation='relu'):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.activation=activation
        self.BN1=nn.BatchNorm2d(in_channels)
        self.BN2=nn.BatchNorm2d(out_channels)
        self.ReLu=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels, out_channels, (3, 3),stride=2,padding=1)
        self.conv2=nn.Conv2d(out_channels, out_channels, (3, 3),stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels, out_channels, (3, 3),stride=2,padding=1)
        
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
        self.maxpool=nn.MaxPool2d(2)
        #self.Conv2=nn.Conv2d(16,32, (3,3))
        self.resblocks1 = nn.Sequential(res_block(16,32))     
        self.resblocks2 = nn.Sequential(res_block(32,32))
        #☺self.Fully1=nn.Linear(32*124*124,16)
        self.Fully1=nn.Linear(32*16*16,16)
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
        img=self.maxpool(img)
        img=self.resblocks1(img)
        img=self.resblocks2(img)
        #img=self.ReLu(img)
        #img=self.Conv2(img)
        #img=img.view(-1, 124 * 124 * 32)
        img=img.view(-1, 16 * 16 * 32)


        img=self.ReLu(img)
        img=self.Fully1(img)
        img=self.ReLu(img) 
        #pdqn)=rint (img)
        wp=self.Fully2(wp)
        wp=self.ReLu(wp)
        wp=self.Fully3(wp)
        wp=self.ReLu(wp)

        print (wp.size())
        print (img.size())
        
        x=torch.cat((img,wp),dim=1)
        
        print (x.size())
        print ('+++++++++')
        
        x=self.Fully4(x)
        
        x=self.ReLu(x) 
        x=self.Fully5(x)
        x=self.ReLu(x) 
        #x=self.dropout(x)
        
        return (x)