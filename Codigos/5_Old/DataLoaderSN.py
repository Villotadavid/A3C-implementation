import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
import numpy as np
import cv2 as cv2
import glob
from torchsummary import summary
import SafeNet
import os
from torchvision import transforms
from PIL import Image


        
def bach_generator (batch_size,directory,index,labels):
    
    frames=glob.glob(directory+'/images/*')
    img = torch.tensor([200,200])
    batch_x = torch.unsqueeze(img,0)
    batch_l = np.zeros(((batch_size),2))
    preprocess = transforms.Compose([
    transforms.Resize((200,200)),
    transforms.ToTensor()])
    
    for i in range(index*batch_size,index*batch_size+batch_size):
        img = Image.open(frames[i])
        img=preprocess (img)
        batch_x[i]=torch.unsqueeze(img,0)
        batch_l[i]=labels[i]
    return(batch_x,batch_l)   

model=SafeNet.Resnet()                  #CPU Model definition

if torch.cuda.is_available():   #Check if GPU is available 
    model.cuda()                #Sends the model to GPU (This avoids variable type errors)
    
summary(model,(3,400, 400))



general='C:/Users/usuario/Desktop/Doctorado/Dataset/Exp51'
frames=glob.glob(general+'/images/*')

X_filename = os.path.join(general, "LabelsX.txt")
Y_filename = os.path.join(general, "LabelsY.txt")
ground_truthX = np.loadtxt(X_filename, usecols=0, skiprows=1)
ground_truthY = np.loadtxt(Y_filename, usecols=0, skiprows=1)
labels=np.zeros((len(ground_truthX ),2)) #[None] * len(ground_truthX )   #Definir vector de ciertalongitud
for i in range (0,len(ground_truthX)):
    labels[i]=[ground_truthX[i],ground_truthY[i]]
n=0
bach_generator(6,general,n,labels)

n_epochs=4
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
sad=[]


for epoch in range(n_epochs):
    for n in range (0,len(frames)):
        x,batch_L=bach_generator(6,general,n,labels)
        outputs = model(x)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
