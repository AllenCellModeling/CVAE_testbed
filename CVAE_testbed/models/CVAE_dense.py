import numpy as np

import torch
import torch.nn as nn

from CVAE_testbed.utils import weight_init
from ..layers import DenseBlock, get_activation


class CVAE(nn.Module):
    # Linear densenet implementation
    def __init__(self, enc_layers, vae_layers, dec_layers, activation_last=None):
        super(CVAE, self).__init__()

        layers_enc = []
        for block in enc_layers:
            layers_enc.append(DenseBlock(block))

        self.encoder_net = nn.Sequential(*layers_enc)

        layers_mu = []
        layers_log_var = []
        for i, block in enumerate(vae_layers):
            if (i + 1) == len(vae_layers):
                layers_mu.append(
                    DenseBlock(block, activation_last=False, bn_last=False)
                )
                layers_log_var.append(
                    DenseBlock(block, activation_last=False, bn_last=False)
                )
            else:
                layers_mu.append(DenseBlock(block))
                layers_log_var.append(DenseBlock(block))

        self.mu = nn.Sequential(*layers_mu)
        self.log_var = nn.Sequential(*layers_log_var)

        layers_dec = []
        for i, block in enumerate(dec_layers):
            if (i + 1) == len(dec_layers):
                layers_dec.append(
                    DenseBlock(block, activation_last=False, bn_last=False)
                )
                layers_dec.append(get_activation(activation_last))
            else:
                layers_dec.append(DenseBlock(block))

        self.decoder_net = nn.Sequential(*layers_dec)

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
