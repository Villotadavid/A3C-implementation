#!/usr/bin/env python

import numpy as np
import cv2 as cv2
import sys
import gflags
import os
import glob
global maxi

def flowToDisplay (flow):

        hsv=np.zeros((480,640,3),dtype=np.float32)	
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        hsv[...,1] = 255
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) 
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        return bgr

##########################################################################
##########################################################################

ResRatio=2
general='C:/Users/usuario/Desktop/Doctorado/Videos/DroneRace_frame'



prev=np.zeros((480,640))
uflow=np.zeros((480,640))
lista = os.listdir(general) # dir is your directory path
number_files = len(lista)
frames=glob.glob(general+'/*')
count=0
mean=0		
	
for fr in range (1,number_files ):

	#frname='frame_{0:05d}'.format(fr)+'.jpg'
	#ofname=destfolder+'/frame_{0:05d}'.format(fr)+'_of.jpg'
	count+=1

	frameRGB= cv2.imread(frames[fr])  
	
	if frameRGB is None:
		print ('ERROR, photo missing n: '+str(fr))

	else:
	
            if fr == 1 :
                h,w= frameRGB.shape[:2]
                frameGray=cv2.cvtColor(frameRGB, cv2.COLOR_BGR2GRAY)
                frameGray=cv2.resize(frameGray,(640,480),1) 
                prev=frameGray
            else:
                
                frameGray=cv2.cvtColor(frameRGB, cv2.COLOR_BGR2GRAY)
                frameGray=cv2.resize(frameGray,(640,480),1) 
                #prev=cv2.resize(prev,(h/ResRatio, w/ResRatio),1) 
                uflow=cv2.calcOpticalFlowFarneback(frameGray,prev, None, 0.5, 3, 15, 3, 5, 1.2, 0);
                uflow=np.array(uflow)
                rgb=flowToDisplay(uflow)
                #mean=mean+np.mean(rgb)
                #rgb[rgb > 1] = 1
                #imageout1=cv2.resize(rgb,(w,h),1) 
                #imageout=imageout1*255
                #print imageout.shape
                cv2.imwrite(str(fr)+'.png',rgb)

            prev=frameGray

mean=mean/count
print ('The mean is: '+str(mean))
#ofname=destfolder+'/frame_'+str(fr+1)+'of.png'
#cv2.imwrite(ofname,imageout)
