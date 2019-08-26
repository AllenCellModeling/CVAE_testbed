import torch
import torch.nn as nn

from CVAE_testbed.utils import weight_init

from ..layers import ResidualLayer


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
                encoder_layers.append(ResidualLayer(i + self.cdim, k))
            #     encoder_layers.append(nn.ReLU())
            elif j == len(enc_layers) - 2:
                self.fc1 = ResidualLayer(i, k, activation=None)
                self.fc2 = ResidualLayer(i, k, activation=None)
            else:
                encoder_layers.append(ResidualLayer(i, k))

        self.encoder_net = nn.Sequential(*encoder_layers)
        # decoder part
        decoder_layers = []
        # decoder_net = nn.Sequential()
        for j, (i, k) in enumerate(zip(dec_layers[0::1], dec_layers[1::1])):
            if j == 0:
                decoder_layers.append(ResidualLayer(i + self.cdim, k))
                # decoder_layers.append(nn.ReLU())
            elif j == len(dec_layers) - 2:
                decoder_layers.append(ResidualLayer(i, k, activation=None))
            else:
                decoder_layers.append(ResidualLayer(i, k))

        self.decoder_net = nn.Sequential(*decoder_layers)

        self.apply(weight_init)

    def encoder(self, x, c):
        concat_input = torch.cat([x, c], 1)
        h = self.encoder_net(concat_input)
        return self.fc1(h), self.fc2(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)  # return z sample

    def decoder(self, z, c):
        concat_input = torch.cat([z, c], 1)
        return self.decoder_net(concat_input)

    def forward(self, x, c):
        mu, log_var = self.encoder(x, c)
        z = self.sampling(mu, log_var)
        return self.decoder(z, c), mu, log_var
