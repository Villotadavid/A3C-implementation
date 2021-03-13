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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #Sirve para silenciarlo warnings de TensorFlow



client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

try:
    responses = client.simGetImages([airsim.ImageRequest("1", 4 ,pixels_as_float = True)]) 
    response=responses[0]
    
    img = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    img=img*255
    
    t, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img, (7, 7), 3)
    t, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow('img',img*255)
    cv2.waitKey()

except KeyboardInterrupt:
    client.reset()
