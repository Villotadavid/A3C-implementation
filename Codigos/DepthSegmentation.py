import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import *
import os
import glob
from datetime import datetime

import numpy as np
import cv2
import glob


general='C:/Users/usuario/Desktop/Doctorado/Codigos/imagenes'
img=general+'/img.png'
img = cv2.imread(img,0)

frame1=np.where(np.logical_and(0<img,img<100))
peso1=np.extract(np.logical_and(0<img,img<100),img)

frame2=np.where(np.logical_and(0<img,img<150))
peso2=np.extract(np.logical_and(0<img,img<150),img)

frame3=np.where(np.logical_and(0<img,img<200))
peso3=np.extract(np.logical_and(0<img,img<200),img)

xc3=int((sum(frame3[0]*peso3))/(sum(peso3)))
yc3=int((sum(frame3[1]*peso3))/(sum(peso3)))

xc2=int((sum(frame2[0]*peso2))/(sum(peso2)))
yc2=int((sum(frame2[1]*peso2))/(sum(peso2)))

xc1=int((sum(frame1[0]*peso1))/(sum(peso1)))
yc1=int((sum(frame1[1]*peso1))/(sum(peso1)))

print(xc1,yc1)



_, thres1 = cv2.threshold(img,100,255,cv2.THRESH_TOZERO)
_, thres2 = cv2.threshold(img,150,200,cv2.THRESH_TOZERO)
_, thres3 = cv2.threshold(img,200,150,cv2.THRESH_TOZERO)

img=cv2.circle(thres1,(yc1,xc1),10,(255,255,0),2)
cv2.imshow('img',thres1)
cv2.waitKey()

img=cv2.circle(thres2,(yc2,xc2),10,(255,255,0),2)
cv2.imshow('img',thres2)
cv2.waitKey()

img=cv2.circle(thres3,(yc3,xc3),10,(255,255,0),2)
cv2.imshow('img',thres3)
cv2.waitKey()


img_ = np.zeros((144,256,3),np.uint8)
img_[36:108,64:192,0]=img[36:108,64:192]
img_[18:126,32:224,1]=img[18:126,32:224]
img_[:,:,2]=img
img=img_

cv2.line(img,(0,72),(256,72),(255,255,0),1)   
cv2.line(img,(128,0),(128,144),(255,255,0),1)

cv2.line(img,(0,144),(128,72),(255,0,0),1)   
cv2.line(img,(256,144),(128,72),(255,0,0),1)

cv2.rectangle(img,(64,36),(192,108),(0,255,0),1)
cv2.rectangle(img,(32,18),(224,126),(0,255,0),1)


cv2.imshow('img',img)
cv2.waitKey()
