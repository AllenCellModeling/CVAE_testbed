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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transforms = transforms.Compose([transforms.ToTensor()])

def idx2onehot(idx, n=15):
    assert idx.shape[1] == 1
    assert torch.max(idx).item() < n + 1

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx.data, 1)

    return onehot

class ConditionLayer(nn.Module):
    def __init__(self, n_in, n_out, n_condition, transpose=False, add_condition=True, activation=True, special=None):

        super().__init__()

        if not transpose:
            self.conv = nn.Conv2d(n_in, n_out, kernel_size = 4, stride = 2, padding=0)
            self.conv_condition = nn.Conv2d(n_condition, n_out, kernel_size = 1, stride = 1)
        else:
            if special is None:
                self.conv = nn.ConvTranspose2d(n_in, n_out, kernel_size = 5, stride = 2, padding = 0)
            else:
                self.conv = nn.ConvTranspose2d(n_in, n_out, kernel_size = 6, stride = 2, padding=1)
            self.conv_condition = nn.ConvTranspose2d(n_condition, n_out, kernel_size = 1, stride = 1)

        if activation:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Sigmoid()

        self.add_condition = add_condition

    def forward(self, x, condition = None):

        x = self.conv(x)

        if self.add_condition and (condition is not None):
            condition = condition.view(condition.shape[0], condition.shape[1], 1, 1)

            condition = self.conv_condition(condition)
            condition = condition.expand(condition.shape[0], condition.shape[1], x.shape[2], x.shape[3])
            x = x + condition

        x = self.activation(x)

        return x

class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, image_channels, n_condition, hidden_dim, latent_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()
        
        self.conv1 = ConditionLayer(image_channels, 64, n_condition)
        self.conv2 = ConditionLayer(64, 128, n_condition)
        self.conv3 = ConditionLayer(128, hidden_dim, n_condition)

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim,latent_dim)

    def forward(self, x, c):
        # x is of shape [batch_size, input_dim + n_classes]

        conv1 = self.conv1(x, c)

        conv2 = self.conv2(conv1, c)

        conv3 = self.conv3(conv2, c)

        hidden = conv3.view(conv3.size(0), -1)



        # latent parameters
        mean = self.mu(hidden)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(hidden)
        # log_var is of shape [batch_size, latent_dim]

        return mean, log_var

class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''
    def __init__(self, image_channels, n_condition, latent_dim, hidden_dim):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.latent_to_hidden = nn.Linear(latent_dim + n_condition, hidden_dim)

        self.conv1 = ConditionLayer(hidden_dim, 128, n_condition, transpose=True)
        self.conv2 = ConditionLayer(128, 64, n_condition, transpose=True)
        self.conv3 = ConditionLayer(64, image_channels, n_condition, transpose=True, activation=False, special=1)

    def forward(self, x, c):
        # x is of shape [batch_size, latent_dim + num_classes]

        x = self.latent_to_hidden(x)

        x = x.view(x.size(0), 1024, 1, 1)

        x = self.conv1(x, c)

        x = self.conv2(x, c)

        generated_x = self.conv3(x, c)


        return generated_x

class CVAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''
    def __init__(self, image_channels, n_condition, hidden_dim, latent_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = Encoder(image_channels, n_condition, hidden_dim, latent_dim)
        self.decoder = Decoder(image_channels, n_condition, latent_dim, hidden_dim)

    def forward(self, x, c):

        # encode
        z_mu, z_var = self.encoder(x, c)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, c), dim=1)
        # decode
        generated_x = self.decoder(z, c)
        return generated_x, z_mu, z_var, x_sample

