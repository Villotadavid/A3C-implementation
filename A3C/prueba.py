from utils import v_wrap, set_init, push_and_pull, record
import os
import signal
import subprocess
import time
import airsim
import numpy as np
from collections import namedtuple

import ImgProc as proc
from Model_A3C import Net
import psutil
import math


client_num=1


sett_dir = 'C:/Users/usuario/Documents/AirSim'

sett_name='/settings'+str(client_num)+'.json'
os.rename(sett_dir+sett_name, sett_dir+'/settings.json')
print ('127.0.0.'+str(client_num+1))
time.sleep(3)
p = subprocess.Popen('C:/Users/usuario/Documents/Forest/Forest.exe', stdout=subprocess.PIPE, shell=True)

time.sleep(10)
os.rename(sett_dir + '/settings.json', sett_dir + sett_name)

for proc in psutil.process_iter():
    try:
        # Get process name & pid from process object.
        processName = proc.name()
        if (processName=='Forest.exe'):

            processID = proc.pid
            print(processName , ' ::: ', processID)
            subprocess.check_output("Taskkill /PID %d /F" % processID)

    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

