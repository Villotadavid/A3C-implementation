import torch
import os
import PIL
import glob
import numpy as np
import cv2 as cv2
import sys
import matplotlib.pyplot as plt

folder='C:/Users/usuario/Desktop/Doctorado/LightRefraction'
img= cv2.imread(folder+'/1.jpg')
img=cv2.resize(img,(567,1008),1) 
print (img.shape)

R1=img[543:562, 328:340]
R2=img[520:537, 284:295]
R3=img[469:483, 222:232]
R4=img[459:473, 202:212]
R5=img[448:461, 183:193]
R6=img[439:452, 170:178]
R7=img[428:440, 157:165]
R8=img[422:435, 143:151]
print(R1.shape)
m=np.zeros(8)

m[0]=R1.mean() #B
m[1]=R2.mean() #N
m[2]=R3.mean() #B
m[3]=R4.mean() #N
m[4]=R5.mean() #B
m[5]=R6.mean() #N
m[6]=R7.mean() #B
m[7]=R8.mean() #N

print (m[0],m[2],m[4],m[6])
print (m[1],m[3],m[5],m[7])

img=cv2.rectangle(img, (328,543), (340,562), (255, 155, 0), 1)
img=cv2.rectangle(img, (284,520), (295,537), (255, 255, 255), 1)
img=cv2.rectangle(img, (222,469), (232,483), (255, 255, 0), 1)
img=cv2.rectangle(img, (202,459), (212,473), (255, 155, 155), 1)
img=cv2.rectangle(img, (183,448), (193,461), (50, 255, 0), 1)
img=cv2.rectangle(img, (170,439), (178,452), (255, 0, 255), 1)
img=cv2.rectangle(img, (157,428), (165,440), (255, 0, 255), 1)
img=cv2.rectangle(img, (143,422), (151,435), (0, 255, 255), 1)

#plt.imshow(im)
#plt.show()
cv2.imshow('image',img)
cv2.waitKey()
