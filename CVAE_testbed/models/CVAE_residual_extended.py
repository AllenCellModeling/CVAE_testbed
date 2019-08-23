import torch
import torch.nn as nn

from CVAE_testbed.utils import weight_init

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# transforms = transforms.Compose([transforms.ToTensor()])

from ..layers import ResidualGroup


class CVAE(nn.Module):
    def __init__(self, enc_layers, vae_layers, dec_layers, activation_last=False):
        super(CVAE, self).__init__()

        self.encoder_net = ResidualGroup(enc_layers)

        self.mu = ResidualGroup(vae_layers, activation_last=False)
        self.log_var = ResidualGroup(vae_layers, activation_last=False)

        self.decoder_net = ResidualGroup(dec_layers, activation_last=activation_last)

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
