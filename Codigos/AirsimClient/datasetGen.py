import airsim
import time
import os
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import os
import numpy as np
import keyboard
import csv


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #Sirve para silenciarlo warnings de TensorFlow
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)                      #Arm or disarm vehicle

path='C:/Users/usuario/Desktop/Doctorado/Codigos'
f1=open(path+"/LabelsX.txt","w")

for idx in range(1,10):
    
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False)]) 
    response=responses[0]
    img = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height,4)
    img=img*255
    frname='frame_{0:05d}'.format(idx)+'.png'
    cv2.imwrite(path+'/imagenes/'+frname, img)
    
    #IMU=client.getImuData())
    IMU = client.getMultirotorState().kinematics_estimated #.linear_acceleration
    Time= client.getMultirotorState().timestamp
    f1.write(frname+','+str(Time)+','+str(IMU.angular_acceleration)+','+str(IMU.angular_velocity)+','+str(IMU.linear_acceleration)+'\n')
f1.close()     
client.enableApiControl(False)
