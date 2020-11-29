import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import *
import os
import glob
from datetime import datetime
import time


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


def data_process(imgR,imgL):

	stereo = cv2.StereoBM_create(32, 25)
	imgL=cv2.resize(imgL,(200,200))
	imgR=cv2.resize(imgR,(200,200))
	disparity = stereo.compute(imgL,imgR)
	#disparity[disparity>100]=255
	#disparity[disparity<100]=0
	
	return(disparity)


def pobj(stereo):

	sub_img1 = np.zeros((200,200,3)) 
        sub_img2 = np.zeros((200,200,3))
	sub_img3 = np.zeros((200,200,3))
	sub_img4 = np.zeros((200,200,3))

        sub_img1 = stereo[0:200,0:200]
	sub_img2 = stereo[200:400,0:200]
	sub_img3 = stereo[0:200,200:400]
	sub_img4 = stereo[200:400,200:400]

	imgfin = np.zeros((400,400,3),np.uint8)

	Vimage=[sub_img1,sub_img2,sub_img3,sub_img4]
	x=[]
	y=[]
	data=[]
	coll=[]
	imgp=[]
	
	for n in range (0,4):
		stereo=Vimage[n]
	
		_, contours, _ = cv2.findContours(np.invert(stereo), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		img = np.zeros((200,200,3))
		#img[:,:,0]=stereo
		#img[:,:,1]=stereo
		img[:,:,2]=stereo			
		maxArea=0
		if len(contours)!=0:
			bop=True
			for c in contours:
    		   	  area = cv2.contourArea(c)
			  if area!=0:
    		   	     if area > maxArea:
				maxArea=area
				IdMaxArea=c
			     
			  else:
			     IdMaxArea=[0,0]
			     bop=False
		else:
			IdMaxArea=[0,0]
			bop=False

		posx,posy,datap=find_center(stereo,IdMaxArea,bop)
       		cv2.drawContours(img, [IdMaxArea], 0, (0, 255, 0), 2, cv2.LINE_AA) 
		
		#cv2.putText(img,str(len(datap)), (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255,2)
		cv2.putText(img,str(n), (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		imgp.append(img)
		collp=float(len(datap))/40000
		x.append(posx)
		y.append(posy)
		data.append(datap)
		coll.append(collp)
	

	imgfin [0:200,0:200]=imgp[0]
	imgfin [200:400,0:200]=imgp[1]
	imgfin [0:200,200:400]=imgp[2]
	imgfin [200:400,200:400]=imgp[3]


	datalist=[[data[0],(x[0],y[0]),coll[0]],[data[2],(x[2]+200,y[2]),coll[2]],[data[1],(x[1],y[1]+200),coll[1]],[data[3],(x[3]+200,y[3]+200),coll[3]]]
	datalist.sort(key=lambda x: x[0].shape,reverse = True) #reverse = True,  #key=lambda x: x.shape
	

	c1=datalist[0][2]
	c2=datalist[1][2]
	c3=datalist[2][2]
	c4=datalist[3][2]


	x1=datalist[0][1][0]
	x2=datalist[1][1][0]
	x3=datalist[2][1][0]

	y1=datalist[0][1][1]
	y2=datalist[1][1][1]
	y3=datalist[2][1][1]

	m1=datalist[0][0].shape[0]
	m2=datalist[1][0].shape[0]
	m3=datalist[2][0].shape[0]


	p1,p2=datalist[0][1]
	p3,p4=datalist[1][1]
	p5,p6=datalist[2][1]

	c1=c1*1.5

	

	if (x1 and y1 and y2 and x3)==50:
		X=200
		Y=200
	else:
		X=(x1*m1*c1+x2*m2*c2+x3*m3*c3)/(m1+m2+m3)
		Y=(y1*m1*c1+y2*m2*c2+y3*m3*c3)/(m1+m2+m3)

	return(int(X),int(Y),p1,p2,p3,p4,p5,p6,imgfin)


def find_center(stereo,contour,bop):
	(xmax,ymax)=stereo.shape
	count=0
	posx=0
	posy=0
	points=[]
	col_x=[]
	col_y=[]
	if bop == True :
	   for x in range (0,xmax):
	      for y in range (0,ymax):
		   if stereo[x,y]==255:                    
			pass#img[x,y,2]=255
		   else:	
		
			if cv2.pointPolygonTest(contour,(y,x), True)>0:		
				points.append((x,y))		#Guarda solo los puntos negros	
				posx=posx+x
				posy=posy+y
				count+=1
			else:
				pass
				#img[x,y,2]=255
	else:
		points=[0]
		posx=0
		posy=0


	points=np.array(points)				#Transforma lista en array

	if count==0:					#En caso de que no haya pixels de colision evita un div por cero
		count=1
	y=posx/count				#Calculo de coordenadas del centroide
	x=posy/count

	return (x,y,points)





def main():
	vectz=10
	p_obj=[0,0]
	datalist=[]
	general='/home/tev/Desktop/Accuracy'
	print general
	Experiments=sorted(glob.glob(general+'/*'))
	number_exp = len(Experiments)
	f=0
	r=70
	print number_exp
	for k in range (0,number_exp):
		general=Experiments[k]
		print general
		frames=sorted(glob.glob( general+'/images/*'))
		number_files = len(frames)
		n=0
		f=f+1
		po=2
		if po==3: #os.path.exists(general+"/LabelsX.txt"):
			print general +' Already processed'
		else:
			'''f1=open(general+"/LabelsX.txt","w")
			f1.write('X Punto objetivo'+'\n')
			f2=open(general+"/LabelsY.txt","w")
			f2.write('Y Punto objetivo'+'\n')'''
			#f3=open(general+"/AccuracyS.txt","w")
			#f3.write('collision points'+'\n')
			path='./Dataset'
			try:
				#os.mkdir(path)
				for fr in range (0,number_files):
	
					stereo = cv2.imread(frames[n],0)
					stereo = cv2.resize(stereo,(400,400))
					t, stereo = cv2.threshold(stereo, 15, 255, cv2.THRESH_BINARY)
					stereo = cv2.GaussianBlur(stereo, (7, 7), 3)
					t, stereo = cv2.threshold(stereo, 0, 255, cv2.THRESH_BINARY)
					

					Xfinal,Yfinal,p1,p2,p3,p4,p5,p6,img=pobj(stereo)
	
					img=cv2.circle(img,(Xfinal,Yfinal),10,(0,209,255),4)
					img=cv2.circle(img,(p1,p2),10,(255,0,0),2)
					img=cv2.circle(img,(p3,p4),10,(255,0,0),2)
					img=cv2.circle(img,(p5,p6),10,(255,0,0),2)

					cv2.imshow('img',img)
					time.sleep(0.5)
					cv2.waitKey(1)	
		
					n=n+1
					p,num=pixeles(stereo,p_obj,r)
					accuracy=num/(len(p)*1.0)

					frname='frame_{0:05d}'.format(n)+'.png'
					cv2.imwrite(path+'/'+frname, img)
					SteerX=Xfinal/400.0
					SteerY=Yfinal/400.0
					#print SteerX,SteerY
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







