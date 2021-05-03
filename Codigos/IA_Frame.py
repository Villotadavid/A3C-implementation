# -*- coding: utf-8 -*-

import setup_path 
import airsim
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import csv

from RandomTrajectory import Trajectory_Generation
import Model
import ImgProc as proc
import ReinforceLearning as RL


import torch
import torch.optim as optim

import torchvision.transforms as T
from PIL import Image




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
        print (self.device)
        self.memory= RL.ReplayMemory(5000)
        self.policy_net = Model.DQN().to(self.device)
        #self.policy_net.load_state_dict(torch.load('C:/Users/usuario/Desktop/Doctorado/Codigos/models/Weights_5460.pt'))
        self.target_net = Model.DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.train()

        self.optimizer=optim.RMSprop(self.policy_net.parameters())


        self.process = T.Compose([T.ToTensor()])
        self.episode_durations = []

    def reset(self):
        self.client.reset()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

    def start(self,trajectory,ax,csvfile,i_episode):

        print("arming the drone...")
        self.client.armDisarm(True)

        landed = self.client.getMultirotorState().landed_state
        if landed == airsim.LandedState.Landed: 
            self.client.takeoffAsync().join()
            
            print ('Despegando...')
            time.sleep(2)
            self.client.moveToPositionAsync(0, 0, -10, 2, 3e+38,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0))
        if plot:
            droneline, =ax.plot3D([0], [0], [0],color='blue',alpha=0.5)
            
        num=0
        done=0
        position=[0,0,0]

            
        for point in trajectory:
                img,state,_,_=proc.get_image(self.client)

                data=self.client.getMultirotorState()

                print('Busqueda del punto',point)
                a=self.client.moveToPositionAsync(int(point[0]), int(point[1]), int(point[2]), 4, 3e+38,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0))

                position=[data.kinematics_estimated.position.x_val ,data.kinematics_estimated.position.y_val,data.kinematics_estimated.position.z_val]
                quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

                Remaining_Length=99

                while not done and Remaining_Length>=2:
                    delta=np.array(point-position,dtype='float32')
                    action,NN_output = RL.select_action(self,state,torch.tensor([delta]))
                    quad_offset=RL.interpret_action(action)
                    self.client.moveByVelocityAsync(quad_vel.x_val,quad_vel.y_val+quad_offset[1],quad_vel.z_val+quad_offset[2], 2)
                    collision_info=self.client.simGetCollisionInfo()
                    reward,Remaining_Length=RL.Compute_reward(img,collision_info,point,position,num)
                    quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

                    #Observe new state

                    img,next_state,_,_=proc.get_image(self.client)
                    data=self.client.getMultirotorState()
                    position=[data.kinematics_estimated.position.x_val ,data.kinematics_estimated.position.y_val,data.kinematics_estimated.position.z_val]
                    done=RL.isDone(reward,collision_info,Remaining_Length)
                    
                    if not done:
                        pass
                    else:
                        next_state = None
                    
                    INDEX=self.memory.push(state,action,next_state,torch.tensor([reward]),torch.tensor([delta]))
                    state = next_state
                    
                    loss,V_Funct,Q_funct,Belleman=RL.optimize_model(self)

                    csvfile.writerow([INDEX,i_episode,action.item(),round(reward,2),round(Remaining_Length,2),round(loss,2),NN_output,point,np.around(position,decimals=2)])
                    if done:
                        #self.episode_durations.append(t+1)
                        print ('Finalizando episodio')
                        break
                                    
                    if plot:
                        position = data.kinematics_estimated.position
                        update_line(droneline,[position.x_val ,position.y_val,-position.z_val])

                num += 1
                if done:
                    break


def train_DQN(nwp,plot):
    

    csvopen=open('Training_data.csv','w',newline='')
    csvfile=csv.writer(csvopen)
    csvfile.writerow(['Index','Episode','Action', 'Reward', 'Remaining length' , 'Loss','NN Output' ,'Punto objetivo','Posición actual'])
    nav = FollowTrajectory()
    
    for i_episode in range(start_episode,start_episode+num_episodes):
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

        nav.start(trajectory,ax,csvfile,i_episode)
        
        if i_episode % TARGET_UPDATE ==0:
                nav.target_net.load_state_dict(nav.policy_net.state_dict())
        nav.reset()

        if i_episode % 10 == 0:
            torch.save(nav.policy_net.state_dict(), Model_PATH+'Weights_'+str(i_episode)+'.pt')
            print ('Saving Model for :'+ str(i_episode)+'th episode')
        

server=1
if server == 1:
    Model_PATH='C:/Users/davillot/Documents/GitHub/Doctorado/Codigos/models'
else:
    Model_PATH='C:/Users/usuario/Desktop/Doctorado/Codigos/models/'
TARGET_UPDATE = 10
num_episodes=20000
start_episode=0

if __name__ == "__main__":  

    parser = argparse.ArgumentParser()
    parser.add_argument("--Plot",help="Get a plot of the trajectories",default=False)
    parser.add_argument("--Waypoints",help="Number of waypoints",default=5)

    args=parser.parse_args()
    
    nwp=args.Waypoints
    plot=args.Plot

    train_DQN(nwp,plot)

