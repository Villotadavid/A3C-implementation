import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import *
import os
import glob
from scipy.optimize import fsolve
from datetime import datetime
import math 

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

	try:
		a,b=solve( ((200-(m*y1+n))**2+(200-(y1))**2)-d**2 ,y1)
		sb=a
		#b-> Dadas la vuelta
		#a-> NOrmal


		if isinstance(sb, complex)==False:
			y1=int(sb[0])
	 		x1=int(y1*m+n)
		else:
			y1=200
			x1=200

	except:
		if m>=1000:
			if x2<=200:

				y1=200
				x1=x2+d
			else :

				y1=200
				x1=x2-d
		elif m==0:
			if x2<=200:

				x1=200
				y1=x2-d
			else :

				x1=200
				y1=x2+d
		else:
			y1=200
			x1=200
	
	return (int(y1),int(x1))




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
			points.append((x,y))
			img[x,y,2]=255		
		else:	
			
			if cv2.pointPolygonTest(contour,(y,x), True)>0:			
				posx=posx+x
				posy=posy+y
				count+=1
			else:
				pass
				#img[x,y,2]=255

	points=np.array(points)				#Transforma lista en array

	if count==0:					#En caso de que no haya pixels de colision evita un div por cero
		count=1
	x=posx/count				#Calculo de coordenadas del centroide
	y=posy/count

	return (x,y,points,img)

def regression_lines(points,stereo): 
#Possible optimization not calling stereo.shape[]

	if points.size!=0:	
										
		[vx,vy,x,y] = cv2.fitLine(points,cv2.DIST_L2,0,0.01,0.01)

		lefty = int((-x*vy/vx) + y)					
		righty = int(((stereo.shape[1]-x)*vy/vx)+y)			
		line1=[(righty,400),(lefty,0)]				
		nx,ny=1,-vx/vy
		mag = np.sqrt((1+ny**2))
		vx2,vy2 = nx/mag,ny/mag
		x2=200
		y2=200

		if vx2==0:
			
			a=9999999
			lefty2 = int((-x2*a) + y2)					
			righty2 = int(((stereo.shape[1]-x2)*a)+y2)
			line2=[(righty2,400),(lefty2,0)]

			p= line_intersection(line1, line2)
			line_params=[0,0,0,0]
 			line_params=[vy/vx,((-x*vy/vx) + y),a,((-x2*a) + y2)]

		elif vx2==1:

			a=0
			lefty2 = int((-x2*a) + y2)					
			righty2 = int(((stereo.shape[1]-x2)*a)+y2)
			line2=[(righty2,400),(lefty2,0)]

			p= line_intersection(line1, line2)
			line_params=[0,0,0,0]
			line_params=[vy/vx,((-x*vy/vx) + y),a,((-x2*a) + y2)]

		else:
			
			lefty2 = int((-x2*vy2/vx2) + y2)					
			righty2 = int(((stereo.shape[1]-x2)*vy2/vx2)+y2)
			line2=[(righty2,400),(lefty2,0)]

			p= line_intersection(line1, line2)
			line_params=[0,0,0,0]
			line_params=[vy/vx,((-x*vy/vx) + y),vy2/vx2,((-x2*vy2/vx2) + y2)]
	else:

		line_params=[0,0,0,0] 
		line1=[(0,200),(400,200)]
		line2=[(400,0),(400,200)]
		p= line_intersection(line1, line2)
	

	if line1[0][0]>=10000:
		line1[0]=(10000,line1[0][1])
	elif line2[0][0]>=10000:
		line2[0]=(10000,line2[0][1])	
	elif line1[0][0]<=-10000:
		line1[0]=(-10000,line1[0][1])
	elif line2[0][0]<=-10000:
		line2[0]=(-10000,line2[0][1])


	if line1[1][0]>=10000:
		line1[1]=(10000,line1[1][1])
	elif line2[1][0]>=10000:
		line2[1]=(10000,line2[1][1])	
	elif line1[1][0]<=-10000:
		line1[1]=(-10000,line1[1][1])
	elif line2[1][0]<=-10000:
		line2[0]=(-10000,line2[0][1])

	return line1,line2,p,line_params

def paint_img(line1,line2,img,p_obj):

	'''img=np.zeros((400,400,1))
	img=cv2.resize(image,(400,400))
	img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) ''' 

	cv2.line(img,(line1[0]),(line1[1]),(255,0,0),2)    #RECTA PARALELA
	cv2.line(img,(line2[0]),(line2[1]),(0,209,255),2)  #RECTA PERPENDICULAR

	cv2.line(img,(0,200),(400,200),(255,255,0),1)   
	cv2.line(img,(200,0),(200,400),(255,255,0),1)   
	
	cv2.circle(img,(p_obj[0],p_obj[1]),10,(0,0,255),2)

	return img

def main():
	vectz=10
	p_obj=[0,0]
	datalist=[]
	general='/home/tev/Desktop/Accuracy'
	Experiments=sorted(glob.glob(general+'/*'))
	print Experiments
	number_exp = len(Experiments)
	f=0
	d=20

	r=70
	for k in range (0,number_exp):
		general=Experiments[k]
		print general
		frames=sorted(glob.glob( general+'/images/*'))
		number_files = len(frames)
		n=0
		f=f+1
		po=2
		if po==3:
			print general +' Already processed'
		else:
			#f1=open(general+"/LabelsX.txt","w")
			#f1.write('X Punto objetivo'+'\n')
			#f2=open(general+"/LabelsY.txt","w")
			#f2.write('Y Punto objetivo'+'\n')
			#f3=open(general+"/AccuracyR.txt","w")
			#f3.write('collision_points'+'\n')
			path='./Dataset/OLD/AlgResult/images_'+str(f)
			x=0
			y=0
			try:
				#os.mkdir(path)
				for fr in range (0,number_files):
					
					stereo = cv2.imread(frames[n],0)
					stereo=cv2.resize(stereo,(400,400))
					t, stereo = cv2.threshold(stereo, 15, 255, cv2.THRESH_BINARY)
					stereo = cv2.GaussianBlur(stereo, (7, 7), 3)
					t, stereo = cv2.threshold(stereo, 0, 255, cv2.THRESH_BINARY)
					# obtener los contornos
					_, contours, _ = cv2.findContours(np.invert(stereo), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
					n=n+1
				
				
					img = np.zeros((400,400,3),np.uint8)
					maxArea=0
					for c in contours:
    				   	   area = cv2.contourArea(c)
    				   	   if area > maxArea:
				      		maxArea=area
				      		IdMaxArea=c
					posx,posy,points,img=find_center(stereo,IdMaxArea,img)
					cv2.drawContours(img, [IdMaxArea], 0, (0, 255, 0), 2, cv2.LINE_AA)
					#print posx,posy
					line1,line2,p,line_params=regression_lines(points,stereo)
					w=200
					#img=stereo
				
					if len(points)!=0 or line1!=[(0,200),(400,200)]:
					
						d=180*0.9*(len(points))/100000
						y1,x1=calcular_distancia (d,line_params[2],line_params[3],p[0],p[1])
					else:
						x1=200
						y1=200

					p_obj=[x1,y1]


					if p_obj[0]>=400:
						p_obj[0]=400
					elif p_obj[0]<=0:
						p_obj[0]=0

					if p_obj[1]>=400:
						p_obj[1]=400
					elif p_obj[1]<=0:
						p_obj[1]=0

					img=paint_img(line1,line2,img,p_obj)
					img=cv2.circle(img,(p_obj[0],p_obj[1]),10,(255,255,255),2)
					cv2.imshow('img',img)
					cv2.waitKey(2)
					frname='framer_{0:05d}'.format(n)+'.png'
					path='/home/tev/Desktop/Dataset/Regresion'
					cv2.imwrite(path+'/'+frname, img)
					p,num=pixeles(stereo,p_obj,r)
					accuracy=num/(len(p)*1.0)
					SteerX=p_obj[0]/400.0
					SteerY=p_obj[1]/400.0
					#f1.write(str(SteerX)+'\n') #str(p_obj[0])+'  ,'+str(p_obj[1])
					#f2.write(str(SteerY)+'\n') #str(p_obj[0])+'  ,'+str(p_obj[1])
					#f3.write(str(accuracy)+'\n')
				#f1.close()
				#f2.close()
				#f3.close()
			except OSError:
	    			print ("Creation of the directory %s failed" % path)

	

		
if __name__ == "__main__":
    main()







