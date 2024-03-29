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

from RandomTrajectory import Trajectory_Generation
import ImgProc as proc
import ReinforceLearning as RL

plt.style.use('ggplot')

def update_line(hl, new_data):
	xdata, ydata, zdata = hl._verts3d
	hl.set_xdata(list(np.append(xdata, new_data[0])))
	hl.set_ydata(list(np.append(ydata, new_data[1])))
	hl.set_3d_properties(list(np.append(zdata, new_data[2])))
	plt.draw()

# Make the drone fly in a circle.
class FollowTrajectory:
    def __init__(self, altitude = -25, speed = 6, snapshots = None):
 
        self.takeoff = False
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.home = self.client.getMultirotorState().kinematics_estimated.position
        print (self.home)


    def start(self,trajectory,ax):
        print("arming the drone...")
        self.client.armDisarm(True)
        
        # AirSim uses NED coordinates so negative axis is up.
        start = self.client.getMultirotorState().kinematics_estimated.position
        landed = self.client.getMultirotorState().landed_state
        if not self.takeoff and landed == airsim.LandedState.Landed: 
            self.takeoff = True
            self.client.takeoffAsync().join()
            start = self.client.getMultirotorState().kinematics_estimated.position

        else:
            print("already flying")

        start = self.client.getMultirotorState()
        print (start)
        
        #self.client.moveToPositionAsync(0, 0, -5, 2).join()
        print ('Drone en posición de inicio')
        position = self.client.getMultirotorState().kinematics_estimated.position
        print (position)
        n=0
        if plot:
            droneline, =ax.plot3D([0], [0], [0],color='blue',alpha=0.5)
        
        for point in trajectory:
            a=self.client.moveToPositionAsync(int(point[0]), int(point[1]), int(point[2]), 2, 3e+38,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0))
            data=self.client.getMultirotorState()
            collision=self.client.simGetCollisionInfo()
            timer=0    
            while not(a._set_flag): # and not(collision.has_collided)

                data=self.client.getMultirotorState()
                collision=self.client.simGetCollisionInfo()
                RL.Compute_reward(collision,data.kinematics_estimated.position,self,point)
                if plot:
                    position = data.kinematics_estimated.position
                    update_line(droneline,[position.x_val ,position.y_val,-position.z_val])
                    plt.pause(0.25)
           

        print("landing...")
        self.client.landAsync().join()

        print("disarming.")
        self.client.armDisarm(False)


    


if __name__ == "__main__":  

    parser = argparse.ArgumentParser()
    parser.add_argument("--Plot",help="Get a plot of the trajectories",default=False)
    parser.add_argument("--Waypoints",help="Number of waypoints",default=6)

    args=parser.parse_args()

    nwp=args.Waypoints
    plot=args.Plot


    nav = FollowTrajectory()
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
