import torch
from Dataset import Dataset,Data_gen
import Model
from torchvision import transforms
from torchsummary import summary
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2



class History(object):
    
    def __init__(self,shape):
        self._buffer=np.zeros(shape,dtype=np.float32)
        self.capacity=capacity
        self.position=0
    
    def value(self):
        return self._buffer
    
    def append (self,state):
        self._buffer[:-1]=self._buffer[1:]  #[:-1] Ommit the last number [1:] Ommit the first number
        self._buffer[-1]=state
        
    def reset (self):
        self._buffer.fill(0)
        
        
        
class DQN_RL_airsim(object):
    def __init__(self,input_shape, nb_actions,
                 gamma=0.99, explorer=0,
                 learning_rate=0.00025, momentum=0.95, minibatch_size=32,
                 memory_size=500000, train_after=10000, train_interval=4, target_update_interval=10000,
                 monitor=True):
        
        self.input_shape = input_shape
        self._history = History(input_shape)
        self.nb_actions = nb_actions
        self.gamma = gamma

        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval

        self._explorer = explorer
        self._minibatch_size = minibatch_size
        self._history = History(input_shape)
        self._memory = ReplayMemory(memory_size, input_shape[1:], 4)
        self._num_actions_taken = 0
        

    
    def get_output(self,actual_image):
        tensor_image=self.img_transform(actual_image)
        self._history.append(tensor_image)
        
        
        
        
        
        
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 5,
          'shuffle': True,
          'num_workers': 0}


''' MODEL GENERATION '''

model=Model.DQN()                  #CPU Model definition

if torch.cuda.is_available():   #Check if GPU is available 
    model.cuda()                #Sends the model to GPU (This avoids variable type errors)
    
n_timesteps=100
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

summary(model,(1,128, 128))
actual_image=cv2.imread('C:/Users/usuario/Desktop/Doctorado/LightRefraction/1.jpg')
print (actual_image.shape)
actual_image=cv2.resize(actual_image,(128,128))
actual_image=cv2.cvtColor(actual_image, cv2.COLOR_BGR2GRAY)
for step in range (n_timesteps):
    actual_image=img_transform(actual_image)
    actual_image=actual_image.to(device)
    action=model(actual_image[None, ...])
    
    
    
    
    
