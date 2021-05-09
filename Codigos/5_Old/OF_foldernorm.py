#!/usr/bin/env python

import numpy as np
import cv2 as cv2
import sys
import gflags
import os
import glob

global mini
mini=0
global maxi
maxi=0


#MAX 43.46

def flowToDisplay (flow,maxi,mini):

	rgb=np.zeros((480,640,1),dtype=np.float32)
	
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	mag=mag/43.464417
	'''dup=[]
	for k in mag:
   		for i in k:
        		dup.append(i)

	if max(dup) > maxi:

		maxi=max(dup)
	else:
		maxi=maxi

	if min(dup) < mini:

		mani=min(dup)
	else:
		mini=mini'''
	rgb=mag
	#hsv = rgb
	#hsv[...,0] = ang*180/np.pi/2
	#hsv[...,1] = 1
	#rgb = cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX) 
	#bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	return [rgb,maxi,mini]

##########################################################################
##########################################################################

ResRatio=4
prefolder='training'
folder='GOPR0278'
imagefolder='/home/tev/Dataset/'+prefolder+'/'+folder+'/images'
destfolder='/home/tev/Dataset/'+prefolder+'/'+folder+'/images_of'
#if not os.path.exists(destfolder):
	#os.makedirs(destfolder)
ofname=''
prev=np.zeros((720,960))
uflow=np.zeros((720,960))

mini=0
maxo=0

list = os.listdir(imagefolder) # dir is your directory path
number_files = len(list)
frames=glob.glob('/home/tev/Dataset/'+prefolder+'/'+folder+'/images/*')
#print frames
maxi=0
mean=0
count=0
for fnumb in frames:
	#print fnumb
	noworth,worth=fnumb.split("frame")
	a=int(filter(str.isdigit, worth))
	if a>maxi:
		maxi=a
	else:
		maxi=maxi
	
#print max

for fr in range (1,maxi ):

	frname='frame_{0:05d}'.format(fr)+'.jpg'
	ofname=destfolder+'/frame_{0:05d}'.format(fr)+'_of.jpg'
	count+=1

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
		[rgb,maxo,mini]=flowToDisplay(uflow,maxo,mini)
		mean=mean+np.mean(rgb)
		rgb[rgb > 1] = 1
		imageout1=cv2.resize(rgb,(w,h),1) 
		imageout=imageout1*255
		#print imageout.shape
		cv2.imwrite(ofname,imageout)
		#cv2.imshow('image',imageout)
		#cv2.waitKey(1)
	prev=frameGray

mean=mean/count
print 'The mean is: '+str(mean)
ofname=destfolder+'/frame_'+str(fr+1)+'of.png'
cv2.imwrite(ofname,imageout)

