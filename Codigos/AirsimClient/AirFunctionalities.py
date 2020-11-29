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

ORIENTATION = airsim.Quaternionr(0.012, -0.156, 0.076, 0.985)

POSITIONS = [(15, 18, -40),  (55, 18, -40), (55, -20, -40), (15, -20, -40), (15, 18, -40)]

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

try:
    if client.getMultirotorState().landed_state == airsim.LandedState.Landed:
        client.takeoffAsync(timeout_sec=5).join()
    client.hoverAsync().join()   
    client.moveToPositionAsync(5,10,40,4)
    time.sleep(100)

    client.landAsync().join()
except KeyboardInterrupt:
    client.reset()
