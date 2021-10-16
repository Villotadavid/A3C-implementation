# -*- coding: utf-8 -*-
import numpy as np
import math
import ImgProc as proc
import airsim
import math
import random

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

################### COMPUTE REWARD ##################################

def Compute_reward(collision_info,position,self,waypoint):      #The position should be the output of the neural network
    Theshold_dist=10
    if collision_info.has_collided:
        reward=-100
    else:
        x,y=proc.Drone_Vision(self)
        dist=math.sqrt(abs(x-position.x_val)**2+abs(x-position.y_val)**2)
        if dist>Theshold_dist:
            reward=-dist/2
        else:
            reward=20
    return (reward)


################### REPLAY MEMORY ##################################
Transition=namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
        self.position=0
        
    def push(self,*args):
        if len(self.memory)<self.capacity:
            self.memory.append(None)
        self.memory[self.position]=Transition(*args)
        self.position=(self.position+1)%self.capacity
        
    def sample(self,batch_size):
        return random.sample(self.memory, batch_sizes)
    
    def __len__(self):
        return len(self.memory)