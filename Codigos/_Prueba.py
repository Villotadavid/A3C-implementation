# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:10:05 2020

@author: usuario
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from RandomTrajectory import Trajectory_Generation


plt.style.use('ggplot')


def update_line(hl, new_data):
	xdata, ydata, zdata = hl._verts3d
	hl.set_xdata(list(np.append(xdata, new_data[0])))
	hl.set_ydata(list(np.append(ydata, new_data[1])))
	hl.set_3d_properties(list(np.append(zdata, new_data[2])))
	plt.draw()


nwp=10
    


t=0
while t<5:
        map = plt.figure()
        map_ax = Axes3D(map)
        map_ax.autoscale(enable=True, axis='both', tight=True)
        
        map_ax.set_xlim3d([-100.0, 100.0])
        map_ax.set_ylim3d([-100.0, 100.0])
        map_ax.set_zlim3d([-100.0, 100.0])
        hl, = map_ax.plot3D([0], [0], [0],color='blue')
        
        trajectory=Trajectory_Generation(nwp,20,-30)
        map_ax.plot3D(trajectory[:,0], trajectory[:,1],trajectory[:,2],color='red')
        
        for point in trajectory:
              print (point)
              update_line(hl, point)
              plt.show(block=False)
              plt.pause(2)
        s=0
