# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:09:06 2020

@author: usuario
"""

import cv2 as cv2
import numpy as np
from scipy.interpolate import interp1d

img=np.zeros((400,600,3),dtype=np.float32)
pts = np.array([[600,400],[0,0]], np.int32)
#pts = pts.reshape((-1,1,2))
x=np.array([300,200,150,100,75,50,20,-0,-50,-100,-125,-150,-175,-200],np.int32)
y1=(x+50)*(x-200)/50
y2=(x-50)*(x-200)/25
y1=y1.astype(int)
y2=y2.astype(int)
xy=np.c_[x,y1]
print (xy)
#Correction,so it starts from the center of the image
x=x+300
y1=y1+400
y2=y2+400
pts1=np.c_[x,y1]
pts2=np.c_[x,y2]
cv2.polylines(img,[pts1],False,(0,255,255))
cv2.polylines(img,[pts2],False,(0,255,255))

cv2.imshow('imagen',img)
cv2.waitKey(5000)
