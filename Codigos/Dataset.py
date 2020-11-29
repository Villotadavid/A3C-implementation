import torch
import os
import PIL
import glob
import numpy as np
from PIL import Image
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, directory):
        'Initialization'
        self.directory=directory
        self.frames=glob.glob(directory+'/images/*')
        self.preprocess = transforms.Compose([
            transforms.Resize((400,400)),
            transforms.ToTensor()])
        
        X_filename = os.path.join(directory, "LabelsX.txt")
        Y_filename = os.path.join(directory, "LabelsY.txt")
        ground_truthX = np.loadtxt(X_filename, usecols=0, skiprows=1)
        ground_truthY = np.loadtxt(Y_filename, usecols=0, skiprows=1)
        
        self.labels={}
        for i in range (0,len(ground_truthX)):
            self.labels[str(i)]=np.array([ground_truthX[i],ground_truthY[i]])
            
  def __len__(self):
        'Denotes the total number of samples'
        return (len(self.frames)-1)

  def __getitem__(self, index):
        'Generates one sample of data'

        img = Image.open(self.frames[index])
        img=self.preprocess (img)
        label= self.labels[str(index)]

        return img,label

def Data_gen(directory,coef):
        general=directory
        X_filename = os.path.join(general, "LabelsX.txt")
        Y_filename = os.path.join(general, "LabelsY.txt")
        ground_truthX = np.loadtxt(X_filename, usecols=0, skiprows=1)
        ground_truthY = np.loadtxt(Y_filename, usecols=0, skiprows=1)
        
        labels={}
        for i in range (0,len(ground_truthX)):
            labels[str(i)]=np.array([ground_truthX[i],ground_truthY[i]])
  
        frames=glob.glob(directory+'/images/*')
        
        Dataset={
                "train":[],
                "validation":[]}
        long=len(frames)
        for i in range(0,int(long*coef)):

            img = Image.open(frames[i])
            Dataset['train'].append(img)
            
        for i in range(0,int(long-long*coef)):

            img = Image.open(frames[i])
            Dataset['validation'].append(img)    
        return Dataset,labels    
        

