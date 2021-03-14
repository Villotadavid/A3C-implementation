import subprocess
import os,signal
import time
import airsim
import shutil

cpu_number=3
client =[None]*cpu_number
sett_dir='C:/Users/usuario/Documents/AirSim'

for client_num in range (0,cpu_number):

    sett_name='/settings'+str(client_num)+'.json'
    print (sett_dir+sett_name)
    os.rename(sett_dir+sett_name, sett_dir+'/settings.json')
    print ('127.0.0.'+str(client_num+1))

    p = subprocess.Popen('C:/Users/usuario/Documents/Forest/Forest.exe')
    time.sleep(8)
    os.rename(sett_dir + '/settings.json', sett_dir + sett_name)

    client[client_num] = airsim.MultirotorClient(ip='127.0.0.'+str(client_num+1))
    client[client_num].confirmConnection()
    client[client_num].enableApiControl(True)





    #p=subprocess.Popen('C:/Users/usuario/Documents/Forest/Forest.exe')
    '''client[client_num] = airsim.MultirotorClient(ip='127.0.0.'+str(client_num))
    client[client_num].confirmConnection()
    client[client_num].enableApiControl(True)'''
