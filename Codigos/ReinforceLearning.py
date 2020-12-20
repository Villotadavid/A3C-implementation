import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import ImgProc as proc
import airsim



def Compute_reward(collision_info,position,self,waypoint):
    Theshold_dist=10
    if collision_info.has_collided:
        reward=-100
    else:
        x,y=proc.Drone_Vision(self)
        dist=math.sqrt(abs(x-position.x_val)**2+abs(x-position.y_val)**2)
        if dist>Theshold_dist:
            reward=-dist/2
        else:
            reward=20
    return (reward)