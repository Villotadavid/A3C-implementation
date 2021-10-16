import torch
from Dataset import Dataset,Data_gen
import SafeNet
from torchvision import transforms
from torchsummary import summary
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 5,
          'shuffle': True,
          'num_workers': 0}

''' DATA GENERATION '''
# Datasets
#partition,labels =Data_gen('C:/Users/usuario/Desktop/Doctorado/Dataset/Exp51',0.7)
Data=Dataset('C:/Users/usuario/Desktop/Doctorado/Dataset/Exp51')


# Generators

training_generator = torch.utils.data.DataLoader(Data, **params)


validation_generator = torch.utils.data.DataLoader(Data, **params)

''' MODEL GENERATION '''

model=SafeNet.Resnet()                  #CPU Model definition

if torch.cuda.is_available():   #Check if GPU is available 
    model.cuda()                #Sends the model to GPU (This avoids variable type errors)
    
summary(model,(1,400, 400))

n_epochs=100
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
sad=[]



for epoch in range(n_epochs):
    for batch, labels in training_generator:
        local_batch, labels = batch.to(device), labels.to(device=device)
        outputs = model(local_batch)

        loss = criterion(outputs, torch.max(labels, 1)[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
