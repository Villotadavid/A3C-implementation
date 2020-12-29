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

################### PARAMETERS ##################################
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

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
    
################### OPTIMIZE MODEL ##################################
        
def optimize_model(memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions=memory.sample(BATCH_SIZE)
    batch=Transition(*zip(*transitions))
    non_final_mask=torch.tensor(tuple(map(lamda s: s is not None,batch.next_state)),device=self.device, dtype=torch.bool)
    non_final_next_states=torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch=torch.cat(batch.state)
    action_batch=torch.at(batch.action)
    reward_batch=torch.cat(batch.reward)
    