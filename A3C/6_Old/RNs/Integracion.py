import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
import numpy as np
import cv2 as cv2
import glob
import DepthNet
import Resnet


general='C:/Users/usuario/Desktop/Doctorado/Codigos/imagenes'
frames=sorted(glob.glob(general+'/*'))


ground_truth = np.loadtxt('C:/Users/usuario/Desktop/Doctorado/Codigos/LabelsX.txt', usecols=0, skiprows=1)


res_block(32,64)

