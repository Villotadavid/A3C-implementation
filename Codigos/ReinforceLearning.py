# -*- coding: utf-8 -*-
import numpy as np
import math
import ImgProc as proc
import airsim
import math
import random
import cv2

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

import Model
import ImgProc as proc


def interpret_action(action):
    print (action)
    scaling_factor = 0.5
    if action == 0:
        quad_offset = (0, -scaling_factor, -scaling_factor)
    elif action == 1:
        quad_offset = (0, 0, -scaling_factor)
    elif action == 2:
        quad_offset = (0, scaling_factor, -scaling_factor)
    elif action == 3:
        quad_offset = (0, -scaling_factor, 0)
    elif action == 4:
        quad_offset = (0, 0, 0)    
    elif action == 5:
        quad_offset = (0, scaling_factor, 0)
    elif action == 6:
        quad_offset = (0, -scaling_factor, scaling_factor)
    elif action == 7:
        quad_offset = (0, 0, scaling_factor)    
    elif action == 8:
        quad_offset = (0, scaling_factor, scaling_factor)


    return quad_offset

################### COMPUTE REWARD ##################################

def Compute_reward(img ,collision_info ,wp2 ,position ):      #The position should be the output of the neural network
    global prev_position
    global prev_reward
    if collision_info.has_collided or position==prev_position:
        R=-10
        L= 999
    else:
        A_l=proc.Drone_Vision(img)                           #A_l Área Libre
        #proc.data_image(action,Correct_action,img)          #Displays objective area
        
        L=math.sqrt((wp2[0]-position[0])*(wp2[0]-position[0])+(wp2[1]-position[1])*(wp2[1]-position[1])+(wp2[2]-position[2])*(wp2[2]-position[2]))
        if L<=2:
            R_l=40
        else:
            
            R_l=R_l=-1*L+40
            
        #R_cp=-10*A_l/(128*128)
        R=R_l #R_cp
        
    '''if prev_reward>R:
        R=R-5
    else:
        R=R+5'''
        
    prev_position=position 
    return R,L


################### REPLAY MEMORY ##################################
Transition=namedtuple('Transition',('state','action','next_state','reward','delta'))

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
        
    def sample(self,batch_sizes):
        return random.sample(self.memory, batch_sizes)
    
    def __len__(self):
        return len(self.memory)
    
##################### SELECT ACTION ##################################
        
def select_action(self,state,wp):
    global steps_done
    sample=random.random()
    eps_threshold=EPS_END+(EPS_START-EPS_END)*math.exp(-1.0*steps_done/EPS_DECAY)
    steps_done+=1
    sample=999
    if sample > eps_threshold:
        with torch.no_grad():

            state, wp = state.to(self.device), wp.to(self.device)
            print(self.policy_net(state, wp / 100))
            return  self.policy_net(state,wp/100).max(1)[1].view(1,1)
    else:
        print('Random')
        return torch.tensor([[random.randrange(9)]],device=self.device,dtype=torch.long)
        
################### OPTIMIZE MODEL ##################################
        
def optimize_model(self):

    if len(self.memory) < BATCH_SIZE:
        return 0
    transitions=self.memory.sample(BATCH_SIZE)
    batch=Transition(*zip(*transitions))
    
    non_final_mask=torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)),device=self.device, dtype=torch.bool)
    non_final_next_states=torch.cat([s for s in batch.next_state if s is not None]) 
    
    non_final_next_delta=torch.cat(batch.delta[0:len(non_final_next_states)])
 
    state_batch=torch.cat(batch.state)
    action_batch=torch.cat(batch.action)
    delta_batch=torch.cat(batch.delta)
    reward_batch=torch.cat(batch.reward)

    ######### Q(s_t) #####################
    state_batch,delta_batch,reward_batch = state_batch.to(self.device),delta_batch.to(self.device),reward_batch.to(self.device)
    state_action_values=self.policy_net(state_batch,delta_batch/100).gather(1,action_batch) #??

    ######### V(s_{t+1}) #####################
    
    next_state_values=torch.zeros(BATCH_SIZE,device=self.device)
    non_final_next_states,non_final_next_delta = non_final_next_states.to(self.device),non_final_next_delta.to(self.device)

    next_state_values[non_final_mask]=self.target_net(non_final_next_states,non_final_next_delta/100).max(1)[0].detach()
    
    ######### Q-values (Belleman equation) #####################
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    ########## Compute Huber loss #######################

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    
    self.optimizer.zero_grad()
    
    loss.backward()  
                 
    n=0
    for param in self.policy_net.parameters():
        n+=1
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()                               
    
    return loss.item()
    
    
def isDone(reward,collision,L):
    done = 0
    if  reward <= -10 or collision.has_collided==True or L>=40:
        done = 1
    return done


 
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 400
TARGET_UPDATE = 10
steps_done=0
prev_position=[0,0,0]
prev_reward=0
