# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 21:13:47 2020

@author: usuario
"""
import numpy as np
import cv2
import airsim

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

def Drone_Vision(png_image):

    t, png_image = cv2.threshold(png_image, 240, 255, cv2.THRESH_BINARY)
    #png_image = cv2.GaussianBlur(png_image, (7, 7), 3)
    #t, png_image = cv2.threshold(png_image, 0, 255, cv2.THRESH_BINARY)
    #cv2.imshow('img',png_image)
    #cv2.waitKey(2)

    contours, _  = cv2.findContours(np.uint8(png_image), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxArea=0

    for c in contours:
        area = cv2.contourArea(c)
        if area > maxArea:
            maxArea=area
            IdMaxArea=c
    posx,posy=find_center(png_image,IdMaxArea)
    #cv2.circle(png_image,(int(posx),int(posy)),10,(255,255,255),2)
	
	
    return (posx,posy)

def get_image(self):
	responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False,False)])
	response = responses[0]
	img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
	img_rgba = img1d.reshape(response.height, response.width, 3)
	img_rgba = cv2.resize(img_rgba, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
	return img_rgba