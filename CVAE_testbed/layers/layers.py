import torch
import torch.nn as nn

import numpy as np


def get_activation(activation):
    if activation == "relu":
        activation = nn.ReLU()
    elif activation == "sigmoid":
        activation = nn.Sigmoid()
    elif (activation is None) or (activation == "none"):
        activation = nn.Sequential()

    return activation


class BasicLayer(nn.Module):
    # takes the input, passes through a layer, and concatenates to the input
    def __init__(self, n_in, n_out, bn=True, activation="relu"):
        super(BasicLayer, self).__init__()

        layers = []

        layers.append(nn.Linear(n_in, n_out)),
        if bn:
            layers.append(nn.BatchNorm1d(n_out))

        layers.append(get_activation(activation))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResidualLayer(nn.Module):
    def __init__(self, n_in, n_out, n_bottleneck=32, bn=True, activation="relu"):
        super(ResidualLayer, self).__init__()

        self.n_in = n_in
        self.n_out = n_out

        self.activation = get_activation(activation)

        main_layer = []
        main_layer.append(nn.Linear(n_in, n_bottleneck))
        if bn:
            main_layer.append(nn.BatchNorm1d(n_bottleneck))

        main_layer.append(nn.ReLU())
        main_layer.append(nn.Linear(n_bottleneck, n_out))

        if bn:
            main_layer.append(nn.BatchNorm1d(num_features=n_out))

        self.residual_block = nn.Sequential(*main_layer)

        if n_in != n_out:
            self.bypass = nn.Linear(n_in, n_out)
        else:
            self.bypass = nn.Sequential()

    def forward(self, x):
        x_out = self.residual_block(x) + self.bypass(x)
        x_out = self.activation(x_out)
        return x_out


class ResidualBlock(nn.Module):
    def __init__(
        self, layers_list, bottleneck_list=None, activation="relu", activation_last=True
    ):
        super(ResidualBlock, self).__init__()

        if bottleneck_list is None:
            bottleneck_list = [32] * (len(layers_list) - 1)

        layers = []

        for i, (n_in, n_out, bottleneck) in enumerate(
            zip(layers_list[0::1], layers_list[1::1], bottleneck_list)
        ):
            if (i + 1 == len(bottleneck_list)) & ~activation_last:
                layers.append(ResidualLayer(n_in, n_out, bottleneck, activation=None))
            else:
                layers.append(
                    ResidualLayer(n_in, n_out, bottleneck, activation=activation)
                )

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ConcatLayer(nn.Module):
    # takes the input, passes through a layer, and concatenates to the input
    def __init__(self, n_in, n_out, bn=True, activation="relu"):
        super(ConcatLayer, self).__init__()

        layers = []

        layers.append(nn.Linear(n_in, n_out)),
        if bn:
            layers.append(nn.BatchNorm1d(n_out))

        layers.append(get_activation(activation))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return torch.cat([self.main(x), x], 1)


class DenseBlock(nn.Module):
    # dense block:  outputs from each layer go to inputs of each subsequent layer, normal linear layer on the last element
    def __init__(
        self, layers_list, activation="relu", activation_last=True, bn_last=True
    ):
        super(DenseBlock, self).__init__()

        if ~activation_last:
            activation_last = None
        else:
            activation_last = activation

        layers = []
        for i, (n_in, n_out) in enumerate(
            zip(np.cumsum(layers_list)[0::1], layers_list[1::1])
        ):

            if i + 1 == (len(layers_list) - 1):
                layers.append(
                    BasicLayer(n_in, n_out, activation=activation_last, bn=bn_last)
                )
            else:
                layers.append(ConcatLayer(n_in, n_out, activation=activation))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
