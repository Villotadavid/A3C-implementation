import airsim
import numpy as np
import os
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    client.takeoffAsync().join()
    time.sleep(2)

client.moveToPositionAsync(5, 5, -10, 4, 3e+38,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()
print ('Adelante')
client.moveByMotorPWMsAsync(1.0,0.3,1.0,0.3,5)
time.sleep(5)

print ('Derecha')
client.moveByMotorPWMsAsync(1.0,0.5,1.0,0.5,5)
time.sleep(5)

print ('Izquierda')
client.moveByMotorPWMsAsync(0.5,1.0,0.5,1.0,5)
time.sleep(5)

print ('Atras')
client.moveByMotorPWMsAsync(1.0,0.5,1.0,0.5,5)
time.sleep(5)