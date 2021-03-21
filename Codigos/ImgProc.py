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

Basic_contours = [np.array([[[0,0]],[[0,42]],[[42,42]],[[42,0]]]),
                  np.array([[[42,0]],[[86,0]],[[86,42]],[[42,42]]]),
                  np.array([[[86,0]],[[128,0]],[[128,42]],[[86,42]]]),
                  np.array([[[0,42]],[[42,42]],[[42,86]],[[0,86]]]),
                  np.array([[[42,42]],[[86,42]],[[86,86]],[[42,86]]]),
                  np.array([[[86,42]],[[128,42]],[[128,86]],[[86,86]]]),
                  np.array([[[0,86]],[[42,86]],[[42,128]],[[0,128]]]),
                  np.array([[[42,86]],[[86,86]],[[86,128]],[[42,128]]]),
                  np.array([[[86,86]],[[128,86]],[[128,128]],[[86,128]]])]
    
def data_image(action=7,correct_point=6,img=np.zeros([128,128],dtype=np.int8)):

    cv2.drawContours(img,Basic_contours,-1,(255), 1, cv2.LINE_AA)
    cv2.drawContours(img,Basic_contours,action,(255), 4, cv2.LINE_AA)
    cv2.drawContours(img,Basic_contours,action,(112), 4, cv2.LINE_AA)
    
    cv2.imshow('Current Frame',img)
    cv2.waitKey(5)
    return 0

def find_center(img, contour):
    (xmax, ymax) = img.shape
    count = 0

    for x in range(0, xmax):
        for y in range(0, ymax):
            if (img[x, y] != 0 and cv2.pointPolygonTest(contour, (y, x), True) >= 0):
                count += 1

        return count


def Drone_Vision(png_image):
    cv2.imshow('Current Frame', png_image)
    cv2.waitKey(1)
    t, png_image = cv2.threshold(png_image, 64, 150, cv2.THRESH_BINARY)             #0=Cercano
    png_image = cv2.GaussianBlur(png_image, (3, 3), 3)                              #255=Lejano
    t, png_image = cv2.threshold(png_image, 0, 200, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        np.uint8(png_image), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 0
    #print (contours)
    if not contours:
        count=0
    else:
        for c in contours:
            area = cv2.contourArea(c)
            if area > maxArea:
                maxArea = area
                IdMaxArea=c
        #count = find_center(png_image, IdMaxArea)

        count= maxArea/(128*128.0)
        img=np.zeros((128,128))
        cv2.drawContours(img, IdMaxArea,-1,255)
        cv2.imshow('Current Frame', img)
        cv2.waitKey(1)
    return count



def get_image(client):
    process = T.Compose([T.ToTensor()])
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis,pixels_as_float=True)])
    response = responses[0]

    try:
        img = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    except:
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, pixels_as_float=True)])
        response = responses[0]
        img = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)

    #img = np.ascontiguousarray(img, dtype=np.float64) / 255
    img = cv2.resize(img, (128, 128))
    # np.array could be dispensable

    return img, process(img/np.max(img)).unsqueeze(0).to('cpu')

'''if __name__=='__main__':
    data_image()'''