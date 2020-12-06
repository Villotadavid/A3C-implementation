#!/usr/bin/env python

import numpy as np
import cv2 as cv2
import sys
import gflags
import os
import glob

def flowToDisplay (flow,maxi):

	rgb=np.zeros((480,640,1),dtype=np.float32)
	#np.max(mag)
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	#print max(mag)
	
	if np.max(mag)> maxi:
		maxi=np.max(mag)
	else:
		maxi=maxi
	print maxi

	mag=mag/70
	rgb=mag
	#print '   frame max: '+str(round(np.max(mag),3))
	#hsv = rgb
	#hsv[...,0] = ang*180/np.pi/2
	#hsv[...,1] = 1
	#rgb = cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX) 
	#bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	print 'Gen max: '+str(round(maxi,3))+'   frame max: '+str(round(np.max(mag),3))
	return [rgb,maxi]

##########################################################################
##########################################################################

ResRatio=2
#prefolder='training'
#folder='DSCN2563'
#imagefolder='/home/tev/Dataset/'+prefolder+'/'+folder+'/images'
#destfolder='/home/tev/Dataset/'+prefolder+'/'+folder+'/images_of'

general='/home/tev/Desktop/LOG/VideosDay-2/5/images'
experiments=glob.glob(general + "/*")
#print(experiments)


frameGray=[]
imageout=[]
imagefolder='/home/tev/Desktop/LOG/VideosDay-2/5/images'
destfolder='/home/tev/Desktop/LOG/VideosDay-2/5/images_of2'
exp='/home/tev/Desktop/LOG/VideosDay-2/5'
ER=False
if not os.path.exists(destfolder):
	os.makedirs(destfolder)
	ofname=''
	prev=np.zeros((720,960))
	uflow=np.zeros((720,960))
	list = os.listdir(imagefolder) # dir is your directory path
	number_files = len(list)
	frames=glob.glob(exp+'/images/*')
	#print frames
	maxi=0
	maxo=0
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
		ofname=destfolder+'/frame_{0:05d}'.format(fr)+'_of.png'
	

		frameRGB= cv2.imread(imagefolder+'/'+frname)  
	
		if frameRGB is None:
			print ('ERROR, photo missing n: '+str(fr))
			ER=True
		else:
	
 		    if fr == 1 :
       			h,w= frameRGB.shape[:2]
			ResRatio=h/720*2
  		    frameGray=cv2.cvtColor(frameRGB, cv2.COLOR_BGR2GRAY)
       		    frameGray=cv2.resize(frameGray,(h/ResRatio, w/ResRatio),1) 
 		    prev=cv2.resize(prev,(h/ResRatio, w/ResRatio),1) 
	
  	 	    if prev.size!=0:
			uflow=cv2.calcOpticalFlowFarneback(prev,frameGray,None,0.4,2,15,3,8,1,0);  #0.4,2,15,15,5,1,0
			uflow=np.array(uflow)
			[rgb,maxo]=flowToDisplay(uflow,maxo)
			#print rgb
			rgb[rgb > 0.95] = 1
			imageout1=cv2.resize(rgb,(w,h),1) 
			imageout=(imageout1*255).astype(dtype=np.uint8)
			if ER==True:
				imageout=prevOF
				ER=False
			else:
				imageout=imageout
				ER=False
			cv2.imwrite(ofname,imageout)
			#cv2.imshow('image',imageout)
			#cv2.waitKey(1)
		prev=frameGray
		prevOF=imageout


	ofname=ofname=destfolder+'/frame_{0:05d}'.format(fr+1)+'_of.png'
	cv2.imwrite(ofname,imageout)
else:
	print('Skipping experiment :'+exp)
		
