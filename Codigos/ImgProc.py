# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 21:13:47 2020

@author: usuario
"""
import numpy as np
import cv2
import airsim
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

def find_center(stereo,contour):
	(xmax,ymax)=stereo.shape
	count=0
	posx=0
	posy=0
	points=[]

	for x in range (0,xmax):
		for y in range (0,ymax):
			if (stereo[x,y]!=255 and cv2.pointPolygonTest(contour,(y,x), True)>=0):
				posx=posx+x
				posy=posy+y
				count+=1
            

	if count==0:					#En caso de que no haya pixels de colision evita un div por cero
		count=1
	y=posx/count				#Calculo de coordenadas del centroide
	x=posy/count

	return (x,y)

def Drone_Vision(self):
    png_image=np.float32(get_image(self))
    t, png_image = cv2.threshold(png_image, 15, 255, cv2.THRESH_BINARY)
    png_image = cv2.GaussianBlur(png_image, (3, 3), 3)
    t, png_image = cv2.threshold(png_image, 0, 255, cv2.THRESH_BINARY)
    contours, _  = cv2.findContours(np.uint8(png_image), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxArea=0
    print (len(contours))
    if not contours:
        posx,posy=64,64
    else:
        for c in contours:
            area = cv2.contourArea(c)
            if area > maxArea:
                maxArea=area
                IdMaxArea=c
        posx,posy=find_center(png_image,IdMaxArea)
	
    return (posx,posy)

def get_image(self):
    responses = self.client.simGetImages([airsim.ImageRequest("1", 4 ,pixels_as_float = True)]) 
    response=responses[0]
    img = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    img=img*255
    img = torch.from_numpy(img)
    
    return self.resize(img).unsqueeze(0).to(self.device)