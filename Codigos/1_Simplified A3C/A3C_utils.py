import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms as T


import os
import subprocess
import time
import airsim
import numpy as np

import psutil
import math

global position
prev_position_1=[0,0,0]
prev_position_2=[0,0,0]
prev_position_3=[0,0,0]

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
        p = subprocess.Popen('C:/Users/davillot/Doctorado/Environments/Forest/Forest/run.bat', stdout=subprocess.PIPE)
    else:
        p = subprocess.Popen('C:/Users/usuario/Documents/Forest/run.bat', stdout=subprocess.PIPE)

    time.sleep(10)
    os.rename(sett_dir + '/settings.json', sett_dir + sett_name)
    time.sleep(2)


def get_PID(PIDs,n):
    PID=0
    for proc in psutil.process_iter():
        try:
            # Get process name & pid from process object.
            processName = proc.name()


            if (processName == 'Forest.exe' and proc.parent() != None):
                    if any(proc.pid == x for x in PIDs):
                        pass
                    else:
                        PID=proc.pid
        except:
            pass
    return PID

def first_start(ip):

    client = airsim.MultirotorClient(ip=ip)
    client.confirmConnection()
    client.enableApiControl(True)
    return client

def client_start(client):

    client.reset()
    client.enableApiControl(True)
    client.takeoffAsync().join()


######################## CLIENT check ########################################

def client_check(threads):

    ip_list=[('127.0.0.' + str(id + 1)) for id in range(0,threads)]
    clients=[airsim.MultirotorClient(ip=ip_list[n]) for n in range(0,threads)]    

######################## CLIENT check ########################################

def loop_check(start_time,l_bool,id,server,PID,MAX,lock):
    time_=0

    while not l_bool:
        pass

    while (l_bool and time_<=MAX+180):
        time_=time.time()-start_time

    with lock:
        if l_bool and time_>=MAX+180:
            print ('Kill: '+str(PID))
            subprocess.check_output("Taskkill /PID %d /F" % PID)
            create_env(id,server)

    return 0
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

def isDone(reward,collision,num):
    done = 0
    if reward <= -1 or collision.has_collided==True or reward==num:
        done = 1
    return done

############################# ACTIONS #############################

def interpret_action(action):

    linear_scaling_factor = 0.75
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
        quad_offset = (-linear_scaling_factor, 0, 0)
    elif action == 6:
        quad_offset = (0, 0, 0)

    return quad_offset

############################# Sync #############################

def check_loop_finish(loop_finish):
    while not all(element for element in loop_finish):
        pass

############################# Compute reward #############################

def guess_stuck (position):
    global prev_position_1
    global prev_position_2
    global prev_position_3

    prev_position_3 = prev_position_2
    prev_position_2 = prev_position_1
    prev_position_1=position

    if prev_position_1==prev_position_3:
        stuck=1
    else:
        stuck=0

    return stuck



def Compute_reward(collision_info ,wp2 ,position,num ):      #The position should be the output of the neural network


    L=math.sqrt((wp2[0]-position[0])*(wp2[0]-position[0])+(wp2[1]-position[1])*(wp2[1]-position[1])+(wp2[2]-position[2])*(wp2[2]-position[2]))
    stuck=guess_stuck (position)
    if collision_info.has_collided or stuck or L>=60:
        R = -1
    elif L >= 40 and L <= 80:
        R = -L / 40 + 1
    else:
        if L <= 1:
            R_l = 1
        else:
            R_l = 0.5 ** (0.15 * L)

        R = R_l
    return R,L