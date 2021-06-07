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
    if collision_info or L>=80:
        R=-1
    elif L>=40 and L<=80:
        R=-L/40+1
    else:
        if L<=1:
            R_l=1
        else:
            R_l=0.5**(0.15*L)
            #R_l=0.5**(0.15*L)
            print (L, R_l)
        R=R_l
     
    
    '''if prev_reward>R:
        R=R-2
    else:
        R=R+2'''
        
    prev_reward=R
    
    return R



if __name__=='__main__':
    
    collision_info=False
    col_prob=0
    L=np.linspace(0,2*math.pi,50)
    sin=np.sin(L)*45+45
    R=[]
    x=[]
    
    n=1
    
    for t in sin:
        R.append(Compute_reward( collision_info, col_prob, t))
        x.append(n)
        n+=1
    
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.plot(x,sin)
    ax2.plot(x,R)
    plt.show()