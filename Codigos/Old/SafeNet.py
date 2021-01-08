import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
import numpy as np
import cv2 as cv2
import glob
from torchsummary import summary

#Pytorch’s LSTM expects all of its inputs to be 3D tensors

#The first axis is the sequence itself
#The second indexes instances in the mini-batch
#The third indexes elements of the input


#Convolutional block specifiation

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
    )

    return block

#Residual block specifiation

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

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv5 = nn.Conv2d(1, 32, kernel_size=5, padding=1,stride=[2,2])
        self.maxpool=nn.MaxPool2d(2)
        self.ReLu=nn.ReLU()
        self.resblocks1 = nn.Sequential(res_block(32,32))     
        self.resblocks2 = nn.Sequential(res_block(32,64))
        self.resblocks3 = nn.Sequential(res_block(64,128))
        self.fc1 = nn.Linear(128 * 13 * 13, 32)
        self.fc2 = nn.Linear(128 * 13 * 13, 2)
        self.dropout = nn.Dropout2d(p=0.5)

        
    def forward(self, x):
        out=self.conv5(x)
        print(out.size())
        out = self.maxpool(out)
        print(out.size())
        out = self.resblocks1(out)
        print(out.size())
        out = self.resblocks2(out)
        out = self.resblocks3(out)
        out = out.view(-1, 128 * 13 * 13)
        out=self.ReLu(out)
        out=self.dropout(out)
        out = self.fc2(out)
        return out

model=Resnet() 
if torch.cuda.is_available():   #Check if GPU is available 
    model.cuda()                #Sends the model to GPU (This avoids variable type errors)
    
summary(model,(1,400,400)) 
#general='C:/Users/usuario/Desktop/Doctorado/Codigos/imagenes'
#frames=sorted(glob.glob(general+'/*'))
#ground_truth = np.loadtxt('C:/Users/usuario/Desktop/Doctorado/Codigos/LabelsX.txt', usecols=0, skiprows=1)
#res_block(32,64)


