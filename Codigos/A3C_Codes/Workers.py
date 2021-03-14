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

import ImgProc as proc
import ReinforceLearning as RL
from Model_A3C import Net

MAX_EP = 3000
MAX_EP_STEP = 200

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

    sett_name='/settings'+str(client_num)+'.json'
    print (sett_dir+sett_name)
    os.rename(sett_dir+sett_name, sett_dir+'/settings.json')
    print ('127.0.0.'+str(client_num+1))

    p = subprocess.Popen('C:/Users/usuario/Documents/Forest/Forest.exe')
    time.sleep(8)
    os.rename(sett_dir + '/settings.json', sett_dir + sett_name)

    client= airsim.MultirotorClient(ip='127.0.0.'+str(client_num+1))
    client.confirmConnection()
    client.enableApiControl(True)

    return client

def reset(self):
        self.client.reset()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

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


def isDone(reward,collision,L):
    done = 0
    if  reward <= -10 or collision.has_collided==True or L>=40:
        done = 1
    return done



class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.id=name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(1,4).double()           # local network
        self.process = T.Compose([T.ToTensor()])
        self.device = torch.device("cpu")
        self.memory = ReplayMemory(5000)
        self.process = T.Compose([T.ToTensor()])

    def run(self):
        print('Still Success')
        self.client = create_env(self.id)
        total_step = 1

        while self.g_ep.value < MAX_EP:

            #Hay que hacer un reset al environment

            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            point = np.empty([3], dtype=np.float32)
            point[0], point[1], point[2] = 30, 15, -20
            img, state = proc.get_image(self)
            data = self.client.getMultirotorState()
            print('Busqueda del punto', point)
            a = self.client.moveToPositionAsync(int(point[0]), int(point[1]), int(point[2]), 4, 3e+38,
                                                airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0))
            position = [data.kinematics_estimated.position.x_val, data.kinematics_estimated.position.y_val,
                        data.kinematics_estimated.position.z_val]
            time.sleep(1.5)
            quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
            Remaining_Length = 99
            print('Estoy aqui4')
            for t in range(MAX_EP_STEP):

                delta = np.array(point - position, dtype='float32')
                action, NN_output = self.lnet.choose_action( state, torch.tensor([delta]))
                #quad_offset = RL.interpret_action(action)
                #self.client.moveByVelocityAsync(quad_vel.x_val, quad_vel.y_val + quad_offset[1],quad_vel.z_val + quad_offset[2], 2)
                collision_info = self.client.simGetCollisionInfo()

                reward, Remaining_Length = RL.Compute_reward(img, collision_info, point, position, 1)
                quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

                # Observe new state

                img, next_state = proc.get_image(self)
                data = self.client.getMultirotorState()
                position = [data.kinematics_estimated.position.x_val, data.kinematics_estimated.position.y_val,
                            data.kinematics_estimated.position.z_val]
                done = isDone(reward, collision_info, Remaining_Length)

                #ep_r += r

                INDEX = self.memory.push(state, action, next_state, torch.tensor([reward]), torch.tensor([delta]))
                state = next_state

                if done:
                    # self.episode_durations.append(t+1)
                    print('Finalizando episodio')
                    break

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    #push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    #buffer_s, buffer_a, buffer_r = [], [], []
                    print ('Update')
                    if done:  # done and print information
                        #record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)

