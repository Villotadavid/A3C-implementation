# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:48:15 2020

@author: David Villota Miranda
"""


from matplotlib import pyplot as plt
import numpy as np
import argparse
import math
'''
parser = argparse.ArgumentParser()
parser.add_argument("--Length",help="Side of the square delimiting the area of the trajectory",default=100)
parser.add_argument("--Waypoints",help="Number of waypoints",default=10)
parser.add_argument("--distance",help="Distance between waypoints",default=20)
parser.add_argument("--Z",help="Maximum height value",default=25)
args=parser.parse_args()

nwp=args.Waypoints+2
D=args.distance
Z=args.Z
'''
def Trajectory_Generation(Waypoints,D,Z):
    nwp=Waypoints+2
    Waypoints=np.empty([nwp,3],dtype=np.float32)
    x,y=0,0
    Waypoints[0]=[0,0,-10]
    for wp in range(1,nwp-1):
        Alpha=np.random.randint(360)
        z=np.random.randint(-Z)
        x=x+D*math.cos(Alpha)
        y=y+D*math.sin(Alpha)
        Waypoints[wp]=[int(x),int(y),int(-z)]
    Waypoints[nwp-1]=[0,0,0]
    #plt.plot(Waypoints[:,0],Waypoints[:,1],'-o')
    #plt.show()
    return (Waypoints)