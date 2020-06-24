import torch.nn as nn
import torch


class AttentionNetwork(nn.Module):

    def __init__(self, shape, activation=nn.ReLU(), bias=True, dropout=None):
        super().__init__()
        self.activation = activation
        layers_in = shape[:-1]
        layers_out = shape[1:]
        net = []

        for j, (f_in, f_out) in enumerate(zip(layers_in, layers_out)):
            net.append(nn.Linear(f_in, f_out, bias=bias))
            if dropout is not None and j is not len(layers_in) - 1:
                net.append(nn.Dropout(dropout))
            if activation is not None and j is not len(layers_in) - 1:
                net.append(activation)

        net.append(nn.Softmax(dim=1))
        self.attention = nn.Sequential(*net)

    def forward(self, inp):
        x = self.attention(inp)
        return x


class NetworkCluster(nn.Module):

    def __init__(self, shape, num_models, activation=nn.ReLU(), bias=True, dropout=None):
        super().__init__()
        self.activation = activation
        self.num_models = num_models
        layers_in = shape[:-1]
        layers_out = shape[1:]
        cluster = []

        for _ in range(num_models):
            net = []
            for j, (f_in, f_out) in enumerate(zip(layers_in, layers_out)):
                net.append(nn.Linear(f_in, f_out, bias=bias))
                if dropout is not None and j is not len(layers_in) - 1:
                    net.append(nn.Dropout(dropout))
                if activation is not None and j is not len(layers_in) - 1:
                    net.append(activation)
            cluster += [net]

        self.cluster = [nn.Sequential(*c) for c in cluster]

    def __getitem__(self, item):
        return self.cluster[item]

    def __len__(self):
        return self.num_models

    def forward(self, inp):
        x = []
        for c in self.cluster:
            x1 = c(inp)
            # for i in range(x1.shape[1]):
            #     if i % 2 == 0:
            #         x1[:, i] = torch.cos(x1[:, i])
            #     else:
            #         x1[:, i] = torch.sin(x1[:, i])
            x += [x1]
        return x
