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




################### COMPUTE REWARD ##################################

def Compute_reward(self,collision_info,img,position):      #The position should be the output of the neural network
    position=position*128
    Theshold_dist=10
    if collision_info.has_collided:
        reward=-100
    else:
        x,y=proc.Drone_Vision(img)
        dist=10
        #dist=math.sqrt(abs(x-position[0].item())**2+abs(x-position[1].item())**2)
        reward=20
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
        
    def sample(self,batch_sizes):
        return random.sample(self.memory, batch_sizes)
    
    def __len__(self):
        return len(self.memory)
    
##################### SELECT ACTION ##################################
        
def select_action(self,state):
    global steps_done
    sample=random.random()
    eps_threshold=EPS_END+(EPS_START-EPS_END)*math.exp(-1.0*steps_done/EPS_DECAY)
    steps_done+=1
    if sample > eps_threshold:
        with torch.no_grad():
            return  policy_net(state).type(torch.long)#.max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(128),random.randrange(128)]],device=device,dtype=torch.long)
        
################### OPTIMIZE MODEL ##################################
        
def optimize_model(self):
    if len(memory) < BATCH_SIZE:
        return
    transitions=memory.sample(BATCH_SIZE)

    batch=Transition(*zip(*transitions))
    print(np.size(batch.state),np.size(batch.action),np.size(batch.reward))
    non_final_mask=torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)),device=device, dtype=torch.bool)
    non_final_next_states=torch.cat([s for s in batch.next_state if s is not None]) 

    state_batch=torch.cat(batch.state)
    action_batch=torch.cat(batch.action)
    reward_batch=torch.cat(batch.reward)

    state_action_values=policy_net(state_batch).gather(1,action_batch)
    
    
    next_state_values=torch.zeros(BATCH_SIZE,device=device)
    print(non_final_next_states.size())
    next_state_values[non_final_mask]=target_net(non_final_next_states) #.max(1)[0].detach()
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
def isDone(reward):
    done = 0
    if  reward <= -10:
        done = 1
    return done
################### PARAMETERS & TRAINING ##################################
class DQN_:
    def __init__(self):

        self.num_episodes = 100


        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

    def train(self,num_episodes):

        for i_episode in range(num_episodes):
            # Initialize the environment and state
            img,state=proc.get_image(self,process,device)

            for t in count():
                # Select and perform an action
                action = select_action(self,state)
                collision_info=self.client.simGetCollisionInfo()

                reward=Compute_reward(self,collision_info,img,action)
                
                #Observe new state
                last_state=state
                img,next_state=proc.get_image(self,process,device)
                memory.push(last_state,action,next_state,torch.tensor([reward]))
                optimize_model(self)
                done=isDone(reward)
                if done:
                    episode_durations.append(t+1)
                    break
                
            if i_episode % TARGET_UPDATE ==0:
                target_net.load_state_dict(policy_net.state_dict())
            self.client.reset()
 
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

steps_done=0
num_episodes=5 


   
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
 
policy_net = Model.DQN().to(device)
target_net = Model.DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer=optim.RMSprop(policy_net.parameters())
memory= ReplayMemory(10000)

process = T.Compose([#T.ToPILImage(),
                    #T.Resize((128,128), interpolation=Image.CUBIC),
                    T.ToTensor()])
episode_durations = []

def main():           

    
    Framework=DQN_()      
    Framework.train(num_episodes)

if __name__=="__main__":
    main()