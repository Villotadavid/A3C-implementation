# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:01:15 2020

@author: usuario
"""


import cv2
vidcap = cv2.VideoCapture('C:/Users/usuario/Desktop/Doctorado/Videos/vtest.avi')
success,image = vidcap.read()
count = 0
while success:
  frname='framer_{0:05d}'.format(count)+'.png'
  cv2.imwrite('C:/Users/usuario/Desktop/Doctorado/Videos/vtest/'+frname, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
