import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms as T

from A3C.pytorch_A3C_F.utils import v_wrap, set_init, push_and_pull, record
import os
import subprocess
import time
import airsim
import numpy as np
from collections import namedtuple

import ImgProc as proc
import ReinforceLearning as RL
from Model_A3C import Net



################### REPLAY MEMORY ##################################
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'delta'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        return self.position

    def sample(self, batch_sizes):
        return random.sample(self.memory, batch_sizes)

    def __len__(self):
        return len(self.memory)


################### CREATE EMVIRONMENTS ##################################

sett_dir='C:/Users/usuario/Documents/AirSim'
def create_env(client_num):
    print (client_num)
    sett_name='/settings'+str(client_num)+'.json'
    print (sett_dir+sett_name)
    os.rename(sett_dir+sett_name, sett_dir+'/settings.json')
    print ('127.0.0.'+str(client_num+1))
    time.sleep(3)
    p = subprocess.Popen('C:/Users/usuario/Documents/Forest/Forest.exe')
    time.sleep(5)
    os.rename(sett_dir + '/settings.json', sett_dir + sett_name)

    client= airsim.MultirotorClient(ip='127.0.0.'+str(client_num+1))

    return client,p

def client_start(client):
    client.reset()
    client.confirmConnection()
    client.enableApiControl(True)
    landed = client.getMultirotorState().landed_state
    if landed == airsim.LandedState.Landed:
        client.takeoffAsync().join()

        print('Despegando...')
        time.sleep(2)




######################## CLIENT RESET ########################################

def reset(self):
        self.client.reset()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)


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
    if  reward <= -10 or collision.has_collided==True or L>=40:
        done = 1
    return done

############################# ACTIONS #############################

def interpret_action(action):

    linear_scaling_factor = 0.65
    angular_scaling_factor = 0.5
    if action == 0:
        angular=False
        quad_offset = (0, 0, +linear_scaling_factor)
    elif action == 1:
        angular = False
        quad_offset = (0, 0, -linear_scaling_factor)
    elif action == 2:
        angular = True
        quad_offset = (0, 0, angular_scaling_factor)
    elif action == 3:
        angular = True
        quad_offset = (0, 0, -angular_scaling_factor)
    elif action == 4:
        angular = True
        quad_offset = (+linear_scaling_factor*3, 0, 0)

    return quad_offset,angular