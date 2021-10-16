# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def get_puntos(img,vector,CoordenadasX,CoordenadasY):
    for t in range (0,len(CoordenadasX)):
        for n in range (0,len(vector)-1):
            vector[t,n]=np.mean(img[CoordenadasY[n],CoordenadasX[t,n]])


    return vector

def print_hi():
    files=glob.glob('C:/Users/usuario/Desktop/Doctorado/Codigos/06_Light/Exp3/*')
    datos=np.zeros(shape=(7,7,len(files)))
    
    CoordenadasY=np.array([350,240,175,125,90,62])
    CoordenadasX=np.array([[5,30,55,82,95,105],[65,83,110,120,130,135],[132,142,153,160,164,168],[200,200,200,200,200,200],[268,258,247,240,236,232],[335,317,290,280,270,265],[395,370,345,328,305,295]])
    n=0
    
    for img in files:
        img=cv.imread(img)
        img = cv.resize(img, (400, 400))
        datos[:,:,n]=get_puntos(img,datos[:,:,n],CoordenadasX,CoordenadasY)
        n+=1
        
    fig,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3)

    ax1.plot((1, 2, 3, 4, 5, 6, 7), datos[:,:,0], color='gray')
    ax2.plot((1, 2, 3, 4, 5, 6, 7), datos[:,:,1], color='blue')
    ax3.plot((1, 2, 3, 4, 5, 6, 7), datos[:,:,2], color='green')
    ax4.plot((1, 2, 3, 4, 5, 6, 7), datos[:,:,3], color='red')
    ax5.plot((1, 2, 3, 4, 5, 6, 7), datos[:,:,4], color='Yellow')
    ax6.plot((1, 2, 3, 4, 5, 6, 7), datos[:,:,5], color='Purple')
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()

