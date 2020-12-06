#!/usr/bin/env python

import numpy as np
import cv2 as cv2
import sys
import gflags
import os
import glob

def flowToDisplay (flow):

	rgb=np.zeros((480,360,3),dtype=np.float32)
	
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv = rgb
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,1] = 1
	hsv[...,2] = cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX) 
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	return bgr

##########################################################################
##########################################################################

ResRatio=4
prefolder='training'
folder='GOPR0278'
imagefolder='/home/tev/Dataset/'+prefolder+'/'+folder+'/images'
destfolder='/home/tev/Dataset/'+prefolder+'/'+folder+'/images_of'
if not os.path.exists(destfolder):
	os.makedirs(destfolder)
ofname=''
prev=np.zeros((720,960))
uflow=np.zeros((720,960))

list = os.listdir(imagefolder) # dir is your directory path
number_files = len(list)
frames=glob.glob('/home/tev/Dataset/'+prefolder+'/'+folder+'/images/*')
#print frames
max=0
for fnumb in frames:
	#print fnumb
	noworth,worth=fnumb.split("frame")
	a=int(filter(str.isdigit, worth))
	if a>max:
		max=a
	else:
		max=max
#print max

for fr in range (1,max ):

	frname='frame_{0:05d}'.format(fr)+'.jpg'
	ofname=destfolder+'/frame_'+str(fr)+'_of.png'
	

	frameRGB= cv2.imread(imagefolder+'/'+frname)  
	
	if frameRGB is None:
		print ('ERROR, photo missing n: '+str(fr))

	else:
	
	     if fr == 1 :
        	h,w= frameRGB.shape[:2]
		ResRatio=h/720*2
	     frameGray=cv2.cvtColor(frameRGB, cv2.COLOR_BGR2GRAY)
             frameGray=cv2.resize(frameGray,(h/ResRatio, w/ResRatio),1) 
	     prev=cv2.resize(prev,(h/ResRatio, w/ResRatio),1) 
	
	     if prev.size!=0:
		uflow=cv2.calcOpticalFlowFarneback(prev,frameGray,None,0.4,1,12,2,8,1,0);
		uflow=np.array(uflow)
		rgb=flowToDisplay(uflow)

		imageout1=cv2.resize(rgb,(w,h),1) 
		imageout=imageout1*255
		print imageout.shape
		cv2.imwrite(ofname,imageout)
		cv2.imshow('image',imageout)
		cv2.waitKey(1)
	prev=frameGray

ofname=destfolder+'/frame_'+str(fr+1)+'of.png'
cv2.imwrite(ofname,imageout)

