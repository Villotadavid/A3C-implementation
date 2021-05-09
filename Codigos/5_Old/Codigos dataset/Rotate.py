import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import *
import os
import glob



def line_intersection(line1, line2):

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y



def calcular_distancia (d,m,n,x2,y2):

	x1=0
	y1=0
	y1 = Symbol('y1')
	a,b=solve( ((m*y1+n-x2)**2+(y1-y2)**2)-d**2 ,y1)  
	y1=int(a[0])
 	x1=m*y1+n

	return (int(y1),int(x1))

def data_process(imgR,imgL):

	stereo = cv2.StereoBM_create(32, 25)
	imgL=cv2.resize(imgL,(400,400))
	imgR=cv2.resize(imgR,(400,400))
	disparity = stereo.compute(imgL,imgR)
	#disparity[disparity>100]=255
	#disparity[disparity<100]=0
	
	return(disparity)

def find_center(stereo):

	(xmax,ymax)=stereo.shape
	count=0
	posx=0
	posy=0
	points=[]

	for i in range (0,xmax):
	   for n in range (0,ymax):
		if stereo[i,n]==255:
			count+=1
			#x2=i-xmax/2
			#y2=ymax/2-n
			points.append((i,n))
			posx=posx+i #x2
			posy=posy+n #y2
		else:
			stereo[i,n]=0

	points=np.array(points)
	if count==0:
		count=1
	posx=posx/count
	posy=posy/count

	return (posx,posy,points)

def regression_lines(points,stereo): 
#Possible optimization not calling stereo.shape[]

	[vx,vy,x,y] = cv2.fitLine(points,cv2.DIST_L2,0,0.01,0.01) 
	
	# Now find two extreme points on the line to draw line
	#y=m*x+n
	lefty = int((-x*vy/vx) + y)
	righty = int(((stereo.shape[1]-x)*vy/vx)+y)

	nx,ny = 1,-vx/vy
	mag = np.sqrt((1+ny**2))
	vx2,vy2 = nx/mag,ny/mag

	## Forcing the line to pass the center 
	x=200
	y=200

	lefty2 = int((-x*vy2/vx2) + y)
	righty2 = int(((stereo.shape[1]-x)*vy2/vx2)+y)

	line_params=[vy/vx,((-x*vy/vx) + y),vy2/vx2,((-x*vy2/vx2) + y)] #[m1,n1,m2,n2]

	line1=[(righty,400),(lefty,0)]
	line2=[(righty2,400),(lefty2,0)]
	p= line_intersection(line1, line2)

	return line1,line2,p,line_params

def paint_img(line1,line2,image,p_obj):

	img=np.zeros((400,400,1))
	img=cv2.resize(image,(400,400))

	img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)    
	cv2.line(img,(line1[0][0],400),(line1[1][0],0),(255,0,0),1)    #RECTA PARALELA
	cv2.line(img,(line2[0][0],400),(line2[1][0],0),(0,255,0),1)  #RECTA PERPENDICULAR

	cv2.line(img,(0,200),(400,200),(255,255,0),1)   
	cv2.line(img,(200,0),(200,400),(255,255,0),1)   
	
	cv2.circle(img,(p_obj[1],p_obj[0]),10,(0,0,255),2)
	#cv2.circle(img,(int(yz),int(xz)),10,(255,0,255),1)
	return img

def main():

	vectz=10
	p_obj=[0,0]
	general='./DATAc_X/validation'
	Experiments=sorted(glob.glob(general+'/*'))
	number_exp = len(Experiments)
	f=0
	for k in range (0,number_exp):
		general=Experiments[k]
		print general
		frames=sorted(glob.glob(general+'/images/*'))
		number_files = len(frames)
		n=0
		f=f+1
		path='./DATAc_X/validation/Expr'+str(f)
		print path
		n=0
		try:
			os.mkdir(path)
			path='./DATAc_X/validation/Expr'+str(f)+'/images'
			os.mkdir(path)
			for fr in range (0,number_files):
		
				img = cv2.imread(frames[n],0)
				img=cv2.rotate(img, cv2.ROTATE_180)
				#print frames[n]
				#cv2.imshow('img',img)
				#cv2.waitKey(2)
				frname='frame_{0:05d}'.format(n)+'.png'
				cv2.imwrite(path+'/'+frname, img)
				n=n+1

		except OSError:
	    		print ("Creation of the directory %s failed" % path)

		
if __name__ == "__main__":
    main()




