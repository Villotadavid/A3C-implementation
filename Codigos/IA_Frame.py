# -*- coding: utf-8 -*-

import setup_path 
import airsim
import os
import sys
import math
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2

from RandomTrajectory import Trajectory_Generation
import Model
import ImgProc as proc
import ReinforceLearning as RL
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import subprocess

#plt.style.use('ggplot')

def update_line(hl, new_data):
	xdata, ydata, zdata = hl._verts3d
	hl.set_xdata(list(np.append(xdata, new_data[0])))
	hl.set_ydata(list(np.append(ydata, new_data[1])))
	hl.set_3d_properties(list(np.append(zdata, new_data[2])))
	#plt.draw()


class FollowTrajectory:
    def __init__(self, altitude = -25, speed = 6, snapshots = None):
 
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
 
        self.memory= RL.ReplayMemory(10000)
        self.policy_net = Model.DQN().to(self.device)
        self.target_net = Model.DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer=optim.RMSprop(self.policy_net.parameters())


        self.process = T.Compose([T.ToTensor()])
        self.episode_durations = []

    def reset(self):
        self.client.reset()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

    def start(self,trajectory,ax,f):

        print("arming the drone...")
        self.client.armDisarm(True)

        landed = self.client.getMultirotorState().landed_state
        if landed == airsim.LandedState.Landed: 
            self.client.takeoffAsync().join()
            
            print ('Despegando...')
            time.sleep(2)
            
        if plot:
            droneline, =ax.plot3D([0], [0], [0],color='blue',alpha=0.5)
            

        done=0
        position=[0,0,0]
        for t in count(): 
            
            for point in trajectory:
                #print (point,position)
                img,state=proc.get_image(self)

                #print (state)
                data=self.client.getMultirotorState()
                f.write('Punto Encontrado\n')
                print('Busqueda del punto',point)
                a=self.client.moveToPositionAsync(int(point[0]), int(point[1]), int(point[2]), 2, 3e+38,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0))
                position=[data.kinematics_estimated.position.x_val ,data.kinematics_estimated.position.y_val,data.kinematics_estimated.position.z_val]
                time.sleep(1)

                Remaining_Length=99
                while not done and Remaining_Length>=2:         
                    delta=np.array(point-position,dtype='float32')
                    action = RL.select_action(self,state,torch.tensor([delta]))
                    quad_offset=RL.interpret_action(action)
                    quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
                    self.client.moveByVelocityAsync(quad_vel.x_val,quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], 2)
                    time.sleep(0.3)
                    collision_info=self.client.simGetCollisionInfo()
                    col_prob=proc.Drone_Vision(img)
                    reward,Remaining_Length=RL.Compute_reward(self,img,collision_info,col_prob,point,position)

                    #Observe new state

                    img,next_state=proc.get_image(self)
                    data=self.client.getMultirotorState()
                    position=[data.kinematics_estimated.position.x_val ,data.kinematics_estimated.position.y_val,data.kinematics_estimated.position.z_val]
                    done=RL.isDone(reward,collision_info,Remaining_Length) 
                    
                    if not done:
                        pass
                    else:
                        #print ('He estado aqui')
                        next_state = None
                    
                    self.memory.push(state,action,next_state,torch.tensor([reward]),torch.tensor([delta]))
                    state = next_state
                    
                    loss=RL.optimize_model(self)
                    f.write(str(action.item())+', '+str(reward)+', '+str(loss)+', '+str(Remaining_Length)+','+str(point)+','+str(position)+'\n')
                    
                    if done:
                        self.episode_durations.append(t+1)
                        print ('Finalizando episodio')
                        break
                                    
                    if plot:
                        position = data.kinematics_estimated.position
                        update_line(droneline,[position.x_val ,position.y_val,-position.z_val])

                        
                if done:
                    break
                    
      
            if done:
                break
  
        
        


def train_DQN(nwp,plot):
    
    f=open('Training data.txt','w+')
    f.write('Action, Reward , Loss, Remaining length, Punto objetivo\n')
    nav = FollowTrajectory()
    
    for i_episode in range(num_episodes):
        f.write('############# EPISODIO '+str(i_episode)+'#################\n')
        print ('########### EPISODIO '+str(i_episode)+' #################')
        trajectory=Trajectory_Generation(nwp,20,-20)

        
        if plot:
            map = plt.figure()
            ax = Axes3D(map)
            ax.autoscale(enable=True, axis='both', tight=True)
            ax.scatter(trajectory[:,0], trajectory[:,1], -trajectory[:,2], cmap='Greens');
            ax.plot(trajectory[:,0], trajectory[:,1], -trajectory[:,2]);
            s=0
            for n in trajectory:
                ax.text(n[0], n[1], -n[2],s)    #Negative due to the SR system
                s+=1
        else:
            ax=0

        nav.start(trajectory,ax,f)
        
        if i_episode % TARGET_UPDATE ==0:
                nav.target_net.load_state_dict(nav.policy_net.state_dict())
        nav.reset()
    f.close()    
        

EPS_DECAY = 200
TARGET_UPDATE = 10


num_episodes=500





if __name__ == "__main__":  

    #Airsim=subprocess.Popen('C:/Users/usuario/Documents/Forest/Forest/Forest.exe',stdout=subprocess.PIPE)
    parser = argparse.ArgumentParser()
    parser.add_argument("--Plot",help="Get a plot of the trajectories",default=False)
    parser.add_argument("--Waypoints",help="Number of waypoints",default=4)

    args=parser.parse_args()
    
    nwp=args.Waypoints
    plot=args.Plot


    
    train_DQN(nwp,plot)
    #Airsim.kill()

