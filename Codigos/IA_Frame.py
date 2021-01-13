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

plt.style.use('ggplot')

def update_line(hl, new_data):
	xdata, ydata, zdata = hl._verts3d
	hl.set_xdata(list(np.append(xdata, new_data[0])))
	hl.set_ydata(list(np.append(ydata, new_data[1])))
	hl.set_3d_properties(list(np.append(zdata, new_data[2])))
	plt.draw()


class FollowTrajectory:
    def __init__(self, altitude = -25, speed = 6, snapshots = None):
 
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")


    def start(self,trajectory,ax):

        print("arming the drone...")
        self.client.armDisarm(True)

        landed = self.client.getMultirotorState().landed_state
        if landed == airsim.LandedState.Landed: 
            self.client.takeoffAsync().join()            
            
        if plot:
            droneline, =ax.plot3D([0], [0], [0],color='blue',alpha=0.5)
            
        
        pt=0
        for point in trajectory:
            
            img,state=proc.get_image(self,process,device)
            data=self.client.getMultirotorState()
            position=[data.kinematics_estimated.position.x_val ,data.kinematics_estimated.position.y_val,-data.kinematics_estimated.position.z_val]
            for t in count():         #loop while moveToPosition finishes and has not collided
                
                action = RL.select_action(self,state)
                quad_offset=RL.interpret_action(action)
                quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
                self.client.moveByVelocityAsync(2, quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], 2).join()
                collision_info=self.client.simGetCollisionInfo()
                col_prob=proc.Drone_Vision(img)
                reward=RL.Compute_reward(self,img,collision_info,col_prob,trajectory[pt+1],position)
                
                #Observe new state
                last_state=state
                img,next_state=proc.get_image(self,process,device)
                data=self.client.getMultirotorState()
                position=[data.kinematics_estimated.position.x_val ,data.kinematics_estimated.position.y_val,-data.kinematics_estimated.position.z_val]
                memory.push(last_state,action,next_state,torch.tensor([reward]))
                RL.optimize_model(self)
                done=RL.isDone(reward,collision_info)  
                

                if done:
                    episode_durations.append(t+1)
                    break
                                    
                if plot:
                    position = data.kinematics_estimated.position
                    update_line(droneline,[position.x_val ,position.y_val,-position.z_val])
                    plt.pause(0.25)
                    
            pt+=1       

           

        print("landing...")
        self.client.landAsync().join()

        print("disarming.")
        self.client.armDisarm(False)


def train_DQN(nav,nwp,plot):
    
    for i_episode in range(num_episodes):
        
        trajectory=Trajectory_Generation(nwp,20,-30)
        
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

        nav.start(trajectory,ax)
        
        
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

steps_done=0
num_episodes=5 


   
use_cuda = torch.cuda.is_available()
device = torch.device("cpu")   #"cuda:0" if use_cuda else "cpu")
 
policy_net = Model.DQN().to(device)
target_net = Model.DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer=optim.RMSprop(policy_net.parameters())
memory= RL.ReplayMemory(10000)

process = T.Compose([T.ToTensor()])
episode_durations = []


if __name__ == "__main__":  

    parser = argparse.ArgumentParser()
    parser.add_argument("--Plot",help="Get a plot of the trajectories",default=False)
    parser.add_argument("--Waypoints",help="Number of waypoints",default=6)

    args=parser.parse_args()

    nwp=args.Waypoints
    plot=args.Plot

    nav = FollowTrajectory()
    
    train_DQN(nav,nwp,plot)


