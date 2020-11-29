import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import *
import os
import glob
from datetime import datetime

def pixeles(img,centro,radio=20):
	n=0
	indices=[]
	(xmax,ymax)=img.shape
	for x in range (0,ymax):
    		for y in range (0,xmax) :
        		dx = x - centro[0];
        		dy = y - centro[1];
        		distanceSquared = dx * dx + dy * dy;
        		if (distanceSquared <= radio**2):
            			indices.append(img[x,y])
				if img[x,y]==255:
					n=n+1
				

	
	return(indices,n)

def find_center(stereo,contour,img):
	(xmax,ymax)=stereo.shape
	count=0
	posx=0
	posy=0
	points=[]
	col_x=[]
	col_y=[]
	for x in range (0,xmax):
	   for y in range (0,ymax):
		if stereo[x,y]==255:                    
			img[x,y,2]=255
		else:	
			
			if cv2.pointPolygonTest(contour,(y,x), True)>0:		
				points.append((x,y))		#Guarda solo los puntos negros	
				posx=posx+x
				posy=posy+y
				count+=1
			else:
				img[x,y,2]=255
				

	points=np.array(points)				#Transforma lista en array

	if count==0:					#En caso de que no haya pixels de colision evita un div por cero
		count=1
	y=posx/count				#Calculo de coordenadas del centroide
	x=posy/count

	return (x,y,points,img)



def main():
	vectz=10
	p_obj=[0,0]
	general='/home/tev/Desktop/Accuracy'
	Experiments=sorted(glob.glob(general+'/*'))
	number_exp = len(Experiments)
	accuracy=[]
	f=12
	
	for k in range (0,number_exp):
		general=Experiments[k]
		print general
		frames=sorted(glob.glob(general+'/images/*'))
		number_files = len(frames)
		n=0
		f=f+1
		r=70
		#f3=open(general+"/Accuracy.txt","w")
		#f3.write('collision points'+'\n')
		'''f1=open(general+"/LabelsX.txt","w")
		f1.write('X Punto objetivo'+'\n')
		f2=open(general+"/LabelsY.txt","w")
		f2.write('Y Punto objetivo'+'\n')'''
		path='./Dataset'
		try:
	    		#os.mkdir(path)
			#os.mkdir(path)
			for fr in range (0,number_files):
		
				stereo = cv2.imread(frames[n],0)
				stereo=cv2.resize(stereo,(400,400))

				t, stereo = cv2.threshold(stereo, 15, 255, cv2.THRESH_BINARY)
				stereo = cv2.GaussianBlur(stereo, (7, 7), 3)
				t, stereo = cv2.threshold(stereo, 0, 255, cv2.THRESH_BINARY)

				# Obtener los contornos
				_, contours, _ = cv2.findContours(np.invert(stereo), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				
                                # Select the biggest area
				img = np.zeros((400,400,3),np.uint8)
				maxArea=0
				for c in contours:
    				   area = cv2.contourArea(c)
    				   if area > maxArea:
				      maxArea=area
				      IdMaxArea=c
				      

				posx,posy,regression_points,img=find_center(stereo,IdMaxArea,img)
       				cv2.drawContours(img, [IdMaxArea], 0, (0, 255, 0), 2, cv2.LINE_AA)   
				p_obj=[posx,posy]
				n=n+1
				p,num=pixeles(stereo,p_obj,r)
				accuracy=num/(len(p)*1.0)
				#.append(collision)
				posx=posx/400.0
				posy=posy/400.0

				#cv2.circle(img,(p_obj[0],p_obj[1]),r,(255,255,255),2)
				cv2.circle(img,(p_obj[0],p_obj[1]),10,(255,255,255),2)
				cv2.imshow('img',img)
				cv2.waitKey(2)
				frname='frame1_{0:05d}'.format(n)+'.png'
				cv2.imwrite(path+'/'+frname, img)
				'''f1.write(str(posx)+'\n') #str(p_obj[0])+'  ,'+str(p_obj[1])
				f2.write(str(posy)+'\n') #str(p_obj[0])+'  ,'+str(p_obj[1])'''
				#f3.write(str(accuracy)+'\n')
			
			'''f1.close()
			f2.close()'''
			#f3.close()
		except OSError:
	    		print ("Creation of the directory %s failed" % path)
		
	
	

		
if __name__ == "__main__":
    main()







