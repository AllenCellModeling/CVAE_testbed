import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, n_bottleneck=32, bn=True, activation="relu"):
        super(ResidualBlock, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif (activation is None) or (activation == "none"):
            self.activation = nn.Sequential()

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


class ResidualGroup(nn.Module):
    def __init__(
        self, layers_list, bottleneck_list=None, activation="relu", activation_last=True
    ):
        super(ResidualGroup, self).__init__()

        if bottleneck_list is None:
            bottleneck_list = [32] * (len(layers_list) - 1)

        layers = list()

        for i, (n_in, n_out, bottleneck) in enumerate(
            zip(layers_list[0::1], layers_list[1::1], bottleneck_list)
        ):

            if (i + 1 == len(bottleneck_list)) & ~activation_last:
                layers.append(ResidualBlock(n_in, n_out, bottleneck, activation=None))
            else:
                layers.append(
                    ResidualBlock(n_in, n_out, bottleneck, activation=activation)
                )

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
