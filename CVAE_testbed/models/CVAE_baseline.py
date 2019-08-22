import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import linalg

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from tqdm import tnrange
from torchsummary import summary


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# transforms = transforms.Compose([transforms.ToTensor()])

class CVAE(nn.Module):
    def __init__(self, x_dim, c_dim, enc_layers, dec_layers):
        super(CVAE, self).__init__()        
        
        self.xdim = x_dim
        self.cdim = c_dim
        self.enc_layer1 = enc_layers[0]
        # encoder part

        encoder_layers = []
        # encoder_net = nn.Sequential() 
        for j, (i, k) in enumerate(zip(enc_layers[0::1], enc_layers[1::1])):
            if j == 0:
                encoder_layers.append(nn.Linear(i + self.cdim, k))
                encoder_layers.append(nn.ReLU())    
            elif j == len(enc_layers)-2: 
                self.fc1 = nn.Linear(enc_layers[-2], enc_layers[-1])
                self.fc2 = nn.Linear(enc_layers[-2], enc_layers[-1])         
            else:
                encoder_layers.append(nn.Linear(i, k))
                encoder_layers.append(nn.ReLU())

        self.encoder_net = nn.Sequential(*encoder_layers)
        # decoder part
        decoder_layers = []   
        # decoder_net = nn.Sequential()  
        for j, (i, k) in enumerate(zip(dec_layers[0::1], dec_layers[1::1])):
            if j == 0:
                decoder_layers.append(nn.Linear(i + self.cdim, k))   
                decoder_layers.append(nn.ReLU())  
            elif j == len(dec_layers) - 2:
                decoder_layers.append(nn.Linear(i, k))
            else:
                decoder_layers.append(nn.Linear(i, k))
                decoder_layers.append(nn.ReLU())
        
        self.decoder_net = nn.Sequential(*decoder_layers)
    
    def encoder(self, x, c):

        concat_input = torch.cat([x, c], 1)
        h = self.encoder_net(concat_input)
        return self.fc1(h), self.fc2(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    def decoder(self, z, c):
        concat_input = torch.cat([z, c], 1)
        return self.decoder_net(concat_input)
    
    def forward(self, x, c):
        mu, log_var = self.encoder(x, c)
        z = self.sampling(mu, log_var)
        return self.decoder(z, c), mu, log_var