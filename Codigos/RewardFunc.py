# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 19:59:17 2021

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import math

prev_reward=0

def Compute_reward(collision_info ,col_prob ,L ):      #The position should be the output of the neural network

    global prev_reward
    if collision_info:
        R=-100
        L= 999
    else:

        #proc.data_image(action,Correct_action,img)          #Displays objective area
        
        #L=math.sqrt((wp2[0]-position[0])*(wp2[0]-position[0])+(wp2[1]-position[1])*(wp2[1]-position[1])+(wp2[2]-position[2])*(wp2[2]-position[2]))
        if L<=2:
            R_l=50
        else:
            
            R_l=-1*L+40
            
        R_cp=-10*col_prob/(128*128)
        R=R_l+R_cp
     
    
    if prev_reward>R:
        R=R-2
    else:
        R=R+2
        
    prev_reward=R
    
    return R



if __name__=='__main__':
    
    collision_info=False
    col_prob=0
    L=np.linspace(0,2*math.pi,50)
    sin=np.sin(L)*25+25
    R=[]
    x=[]
    
    n=1
    
    for t in sin:
        R.append(Compute_reward( collision_info, col_prob, t))
        x.append(n/2)
        n+=1
    
    fig,ax = plt.subplots()
    ax.plot(x,sin)
    ax.plot(x,R)
    plt.show()