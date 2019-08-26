import numpy as np

import torch
import torch.nn as nn

from CVAE_testbed.utils import weight_init
from ..layers import DenseBlock, get_activation


class CVAE(nn.Module):
    # Linear densenet implementation
    def __init__(self, enc_layers, vae_layers, dec_layers, activation_last=None):
        super(CVAE, self).__init__()

        self.encoder_net = DenseBlock(enc_layers)

        n_enc = np.sum(enc_layers)
        n_vae = np.sum(vae_layers)
        n_dec = np.sum(dec_layers[0:-1])

        self.mu = nn.Sequential(
            DenseBlock([n_enc] + vae_layers), nn.Linear(n_enc + n_vae, vae_layers[-1])
        )
        self.log_var = nn.Sequential(
            DenseBlock([n_enc] + vae_layers), nn.Linear(n_enc + n_vae, vae_layers[-1])
        )

        self.decoder_net = nn.Sequential(
            DenseBlock(dec_layers[0:-1]),
            nn.Linear(n_dec, dec_layers[-1], vae_layers[-1]),
            get_activation(activation_last),
        )

        self.apply(weight_init)

    def encoder(self, x, c):
        concat_input = torch.cat([x, c], 1)
        h = self.encoder_net(concat_input)
        return self.mu(h), self.log_var(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)  # return z sample

    def decoder(self, z, c):
        concat_input = torch.cat([z, c], 1)
        x_hat = self.decoder_net(concat_input)
        return x_hat

    def forward(self, x, c):
        mu, log_var = self.encoder(x, c)
        z = self.sampling(mu, log_var)
        x_hat = self.decoder(z, c), mu, log_var
        return x_hat
