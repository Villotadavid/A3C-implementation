import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms as T

from utils import v_wrap, set_init, push_and_pull, record
import os
import subprocess
import time
import airsim
import numpy as np
from collections import namedtuple

import ImgProc as proc
import ReinforceLearning as RL
from Model_A3C import Net
import psutil
import math


################### REPLAY MEMORY ##################################

global position
prev_position=[0,0,0]

class memory():
    def __init__(self, capacity):
        self.capacity=15000
        self.values=[]
        self.log_probs=[]
        self.entropies=[]

    def push (self,value,log_prob,reward,entropy):

        if len(a) < self.capacity:

            self.values.append(None)
            self.log_probs.append(None)
            self.rewards.append(None)
            self.entropies.append(None)

            self.values[position] = value
            self.log_probs[position] =log_prob
            self.rewards[position] =reward
            self.entropies[position] = entropy

        else:
            del values[0]
            del log_probs[0]
            del rewards[0]
            del entropies[0]
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.entropies.append(entropy)



################### CREATE EMVIRONMENTS ##################################

def create_env(client_num,server):


    if server:
        sett_dir = 'C:/Users/davillot/Documents/AirSim'
    else:
        sett_dir = 'C:/Users/usuario/Documents/AirSim'

    sett_name='/settings'+str(client_num)+'.json'
    os.rename(sett_dir+sett_name, sett_dir+'/settings.json')
    print ('127.0.0.'+str(client_num+1))
    time.sleep(3)
    if server:
        p = subprocess.Popen('C:/Users/davillot/Doctorado/Environments/Forest/Forest/run.bat')
    else:
        p = subprocess.Popen('C:/Users/usuario/Documents/Forest/Forest.exe', stdout=subprocess.PIPE)

    time.sleep(10)
    os.rename(sett_dir + '/settings.json', sett_dir + sett_name)


def client_start(ip):
    client = airsim.MultirotorClient(ip=ip)
    client.reset()
    client.confirmConnection()
    client.enableApiControl(True)
    landed = client.getMultirotorState().landed_state
    if landed == airsim.LandedState.Landed:
        client.takeoffAsync().join()
    return client

######################## CLIENT RESET ########################################

def reset(ip):

        client = airsim.MultirotorClient(ip=ip)
        client.reset()
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)


############################# IA PARAMETERS #############################

# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).expand_as(out)) # thanks to this initialization, we have var(out) = std^2 AQUI HABÍA UN .sum(1)

    return out

# Initializing the weights of the neural network in an optimal way for the learning
def weights_init(m):
    classname = m.__class__.__name__ # python trick that will look for the type of connection in the object "m" (convolution or full connection)
    if classname.find('Conv') != -1: # if the connection is a convolution
        weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
        fan_in = np.prod(weight_shape[1:4]) # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros
    elif classname.find('Linear') != -1: # if the connection is a full connection
        weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
        fan_in = weight_shape[1] # dim1
        fan_out = weight_shape[0] # dim0
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros

############################# DONE #############################

def isDone(reward,collision,L):
    done = 0
    if reward <= -10 or collision.has_collided==True or reward == 50:
        done = 1
    return done

############################# ACTIONS #############################

def interpret_action(action):

    linear_scaling_factor = 4
    if action == 0:
        quad_offset = (0, 0, +linear_scaling_factor)
    elif action == 1:
        quad_offset = (0, 0, -linear_scaling_factor)
    elif action == 2:
        quad_offset = (0, linear_scaling_factor, 0)
    elif action == 3:
        quad_offset = (0, -linear_scaling_factor, 0)
    elif action == 4:
        quad_offset = (+linear_scaling_factor, 0, 0)
    elif action == 5:
        quad_offset = (0, 0, 0)

    return quad_offset

############################# Sync #############################

def check_loop_finish(loop_finish):
    while not all(element for element in loop_finish):
        pass

############################# Compute reward #############################


def Compute_reward(img ,collision_info ,wp2 ,position,num ):      #The position should be the output of the neural network
    global prev_position
    num=0
    resta=np.array(position)-np.array(prev_position)
    dist=0.1
    diff=np.array([dist,dist,dist])
    L=math.sqrt((wp2[0]-position[0])*(wp2[0]-position[0])+(wp2[1]-position[1])*(wp2[1]-position[1])+(wp2[2]-position[2])*(wp2[2]-position[2]))
    
    if collision_info.has_collided or position==prev_position or L>=80:
        R=-10
        L= 999
    else:    
        if L<=2:
            R_l=50
        else:
            R_l=-0.5*L+40

        R=R_l #+num*50 #+R_c
        
    prev_position=position
    
    return R,L