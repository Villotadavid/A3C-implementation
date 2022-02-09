#!/usr/bin/env python

import numpy as np
import cv2 as cv2
import sys
import gflags
import os
import glob
global maxi

def flowToDisplay (flow,maxi):

	rgb=np.zeros((480,640,1),dtype=np.float32)
	
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	#print max(mag)
	mag=mag/140
	
	rgb=mag
	if np.max(mag)> maxi:
		maxi=np.max(mag)
	else:
		maxi=maxi
	print maxi

	print 'Gen max: '+str(round(maxi,3))+'   frame max: '+str(round(np.max(mag),3))
	#hsv = rgb
	#hsv[...,0] = ang*180/np.pi/2
	hsv[..., 2] = 1
	#hsv[...,2] = 1
	#rgb = cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX) 
	#bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	return [rgb,maxi]

##########################################################################
##########################################################################

ResRatio=2
#prefolder='training'
#folder='DSCN2563'
#imagefolder='/home/tev/Dataset/'+prefolder+'/'+folder+'/images'
#destfolder='/home/tev/Dataset/'+prefolder+'/'+folder+'/images_of'

general='/home/tev/Dataset/testing'
experiments=glob.glob(general + "/*")
#print(experiments)

for exp in experiments:
	print ('Optical Flow computing in '+ exp)
	imagefolder=exp+'/images'
	destfolder=exp+'/images_of2'
	maxi=0
	if not os.path.exists(destfolder):
		os.makedirs(destfolder)
	
		ofname=''
		prev=np.zeros((480,640))
		uflow=np.zeros((480,640))

		lista = os.listdir(imagefolder) # dir is your directory path
		number_files = len(lista)

		frames=sorted(glob.glob(exp+'/images/*'))
		
	
		for fr in range (0,number_files+1):
		
			noworth,worth=frames[fr].split("images")
			a=int(filter(str.isdigit, worth))
			
			
			ofname=destfolder+'/'+str(a)+'.png'
			print ofname

			frameRGB= cv2.imread(frames[fr])  
			#print(frameRGB)
			if frameRGB is None:
				print ('ERROR, photo missing n: '+str(fr))
	
			else:
		
	 	 	   if fr == 0 :
        			h,w= frameRGB.shape[:2]
				#ResRatio=h/720*2
	  		   frameGray=cv2.cvtColor(frameRGB, cv2.COLOR_BGR2GRAY)
         	  	  #frameGray=cv2.resize(frameGray,(h/ResRatio, w/ResRatio),1) 
		  	  #print(frameGray)
	 	  	  #prev=cv2.resize(prev,(h/ResRatio, w/ResRatio),1) 
	
	  	  	   if prev.size!=0:
				uflow=cv2.calcOpticalFlowFarneback(prev,frameGray,None,0.4,2,15,3,8,1,0);    #(prev,frameGray,None,0.4,1,12,2,8,1.2,0);
				uflow=np.array(uflow)
				[rgb,maxi]=flowToDisplay(uflow,maxi)

				#imageout1=cv2.resize(rgb,(w,h),1) 
				rgb[rgb > 1] = 1
				imageout1=cv2.resize(rgb,(w,h),1) 
				imageout=(imageout1*255).astype(dtype=np.uint8)
				#print 'hola'
				cv2.imwrite(ofname,imageout)
				#cv2.imshow('image',imageout)
				#cv2.waitKey(1)
			prev=frameGray

		#ofname=destfolder+'/'+str(a+1)+'_of.png'
		#cv2.imwrite(ofname,imageout)
	else:
		print('Skipping experiment :'+exp)
