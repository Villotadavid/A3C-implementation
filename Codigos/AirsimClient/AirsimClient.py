import setup_path 
import airsim
import time
import os
import numpy as np
import cv2
import glob
from tensorflow import keras
from keras import backend as K
from keras import models
from keras.models import Sequential
from matplotlib import pyplot as plt
import utils
import os
import numpy as np
import keyboard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# connect to the AirSim simulator 
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

#car_controls.brake = 1
#car_controls.steering = 0
#client.setCarControls(car_controls)

model1 = utils.jsonToModel('./model_struct_X.json')
model2 = utils.jsonToModel('./model_struct_Y.json')
# Load weights
model1.load_weights('./weights_X.h5')
model2.load_weights('./weights_Y.h5')
print("Loaded model from {}".format('./weights_X.h5'))

model1.compile(loss='mse', optimizer='sgd')
model2.compile(loss='mse', optimizer='sgd')

client.takeoffAsync()
time.sleep(2)
x1=0
y1=0

collision_info = client.simGetCollisionInfo()
datos=client.getImuData()
print (type(datos))

print (collision_info.has_collided)

while collision_info.has_collided==False:

    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized, True)]) 
    response=responses[0]
    img = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    img=img*1020
    a, img = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    coll=cv2.countNonZero(img)
    img = utils.callback_img(img, (320,240), (200,200),0,0)
    outs1 = model1.predict_on_batch(img[None])
    outs2 = model2.predict_on_batch(img[None])
    x=outs1
    y=outs2
    Vxz=int((x-0.5)*11)
    Vy=int((y-0.5)*11)
    Vx=(((40000-coll)/40000)*3)
    client.moveByVelocityAsync(Vx, Vy, Vxz, 5, airsim.DrivetrainType.ForwardOnly)
    print (x, y, coll)
    print (Vx, Vy, Vxz)
    print ('============')
    x1=x
    y1=y
    img=cv2.circle(img,(x*200,y*200),10,(255,255,255),2)
    cv2.imshow('img',img)
    cv2.waitKey(1)

    #frname='frame_{0:05d}'.format(n)+'.png'
    #cv2.imwrite(frname, img)


#restore to original state
client.reset()

client.enableApiControl(False)


            

