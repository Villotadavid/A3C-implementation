#!/usr/bin/env python
# coding: utf-8

# demo

"""
Author: Ke Xian
Email: kexian@hust.edu.cn
Create_Date: 2019/05/21
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

import os, argparse, sys
import numpy as np
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import warnings
warnings.filterwarnings("ignore")
from PIL import Image

sys.path.append('C:/Users/usuario/Desktop/Doctorado/Depth perception/Exial/models')
import DepthNet

# =======================
# demo
# =======================
def IA_process(net, img,n):
        img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        #print (img.shape)
        tensor_img = img_transform(img)
        #print (tensor_img.size())
        # forward
        input_img = torch.autograd.Variable(tensor_img.cuda().unsqueeze(0), volatile=True)  #
        
        output = net(input_img)

        # Normalization and save results
        depth = output.squeeze().cpu().data.numpy()
        min_d, max_d = depth.min(), depth.max()
        depth_norm = (depth - min_d) / (max_d - min_d) * 255
        depth_norm = depth_norm.astype(np.uint8)
        image_pil = Image.fromarray(depth_norm)
        #plt.imsave('C:/Users/usuario/Desktop/Doctorado/Codigos/imagenes/img'+str(n)+'.png', np.asarray(image_pil), cmap='inferno')
        return image_pil 


def IA_setup (gpu_id=0):



    gpu_id = gpu_id
    torch.cuda.device(gpu_id)

    net = DepthNet.DepthNet()
    net = torch.nn.DataParallel(net, device_ids=[0]).cuda()     #Implements data parallelism at the module level. Popular technique used to speed
                                                                #up training on large mini-batches when each mini-batch is too large to fit on a GPU
    checkpoint = torch.load('model.pth.tar')                    #Recoge el directorio del checkpoint. (Los pesos)                  
    net.load_state_dict(checkpoint['state_dict'])               #Loads them into the net
    net.eval()

    return net
