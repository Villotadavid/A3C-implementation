import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms as T
import numpy as np

# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).expand_as(out)) # thanks to this initialization, we have var(out) = std^2
    return out

# Initializing the weights of the neural network in an optimal way for the learning
def weights_init(m):
    classname = m.__class__.__name__ # python trick that will look for the type of connection in the object "m" (convolution or full connection)
    if classname.find('Conv') != -1: # if the connection is a convolution
        weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
        fan_in = np.prod(weight_shape[1:4]) # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros
    elif classname.find('Linear') != -1: # if the connection is a full connection
        weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
        fan_in = weight_shape[1] # dim1
        fan_out = weight_shape[0] # dim0
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros

# Making the A3C brain
class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__() 

        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=4, padding=1)
        self.downsample = nn.Sequential(
                nn.Conv2d(1, 32, 3,  stride=16, padding=1),
                nn.BatchNorm2d(32)
            )
        self.ReLu = nn.ReLU()

    def forward(self, x):
        
        identity = x
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        identity = self.downsample(identity)
        x += identity
        x = self.ReLu(x)
        return x


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Net, self).__init__()

        self.Normalization = nn.BatchNorm2d([1,3,128,128])
        self.ReLu = nn.ReLU()
        self.Resn=ResidualBlock()
        self.ResOF=ResidualBlock()
        self.Fully2=nn.Linear(3,16)
        self.Fully4 = nn.Linear(32, 9)
        self.Fully1=nn.Linear(32*8*8,16)
        self.lstm = nn.LSTMCell(48, 256)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)

        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.distribution = torch.distributions.Normal

        self.train()

    def forward(self, inputs):

        img,ImgOF,delta, (hx, cx) = inputs
        (hx, cx)=(hx.double(), cx.double())

        img = img.double()
        imgOF = img.double()
        delta=delta.double()

        x=self.Resn(img)
        xOF=self.ResOF(imgOF)

        x = x.view(-1, 32 * 8 * 8)
        xOF = xOF.view(-1, 32 * 8 * 8)

        delta=self.Fully2(delta)
        delta=self.ReLu(delta)

        x=self.Fully1(x)
        x=self.ReLu(x)

        xOF=self.Fully1(xOF)
        xOF=self.ReLu(xOF)

        x = torch.cat((x,xOF,delta), dim=1)
        hx, cx = self.lstm(x,(hx,cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss