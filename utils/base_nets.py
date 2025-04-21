import torch.nn as nn
import torch
import functools
import torch
import torch.nn as nn
import math
import numpy as np


def init_weights_requ(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # if hasattr(m, 'bias'):
        #     nn.init.uniform_(m.bias, -.5,.5)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_out')
        if hasattr(m, 'bias'):
            nn.init.uniform_(m.bias, -1, 1)
            # m.bias.data.fill_(0.)


def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))
        # if hasattr(m, 'bias'):
        #     m.bias.data.fill_(0.)


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))
        # if hasattr(m, 'bias'):
        #     m.bias.data.fill_(0.)


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class FirstSine(nn.Module):
    def __init__(self, w0=20):
        """
        Initialization of the first sine nonlinearity.

        Parameters
        ----------
        w0: float
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0 * input)


class Sine(nn.Module):
    def __init__(self, w0=20.0):
        """
        Initialization of sine nonlinearity.

        Parameters
        ----------
        w0: float
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0 * input)


class RandSine(nn.Module):
    def __init__(self, mu_w0=50, std_w0=40, num_features=256):  # 30, 29
        super().__init__()
        self.w0 = mu_w0 + 2. * std_w0 * (torch.rand(num_features, dtype=torch.float32) - .5).cuda()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.w0 * input)


class ReQU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace)

    def forward(self, input):
        # return torch.sin(np.sqrt(256)*input)
        return .5 * self.relu(input) ** 2


class MSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.cst = torch.log(torch.tensor(2.))

    def forward(self, input):
        return self.softplus(input) - self.cst


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class ReQLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_sq = 1 ** 2

    def forward(self, input):
        r_input = torch.relu(input)
        return self.p_sq * (torch.sqrt(1. + r_input ** 2 / self.p_sq) - 1.)


def layer_factory(layer_type):
    layer_dict = \
        {'relu': (nn.ReLU(inplace=True), init_weights_normal),
         'requ': (ReQU(inplace=False), init_weights_requ),
         'reqlu': (ReQLU, init_weights_normal),
         'sigmoid': (nn.Sigmoid(), init_weights_xavier),
         'fsine': (Sine(), first_layer_sine_init),
         'sine': (Sine(), sine_init),
         'randsine': (RandSine(), sine_init),
         'tanh': (nn.Tanh(), init_weights_xavier),
         'selu': (nn.SELU(inplace=True), init_weights_selu),
         'gelu': (nn.GELU(), init_weights_selu),
         'swish': (Swish(), init_weights_selu),
         'softplus': (nn.Softplus(), init_weights_normal),
         'msoftplus': (MSoftplus(), init_weights_normal),
         'elu': (nn.ELU(), init_weights_elu)
         }
    return layer_dict[layer_type]


class FCBlock(nn.Module):
    def __init__(self, in_features, features, out_features,
                 nonlinearity='relu', last_nonlinearity=None,
                 batch_norm=False, group_norm=0, zero_init=False, dropout=0.0):
        """
        Initialization of a fully connected network.

        Parameters
        ----------
        in_features: int
        features: list
        out_features: int
        nonlinearity: str
        last_nonlinearity: str
        batch_norm: bool
        """
        super().__init__()

        # Create hidden features list
        self.hidden_features = [int(in_features)]
        if features != []:
            self.hidden_features.extend(features)
        self.hidden_features.append(int(out_features))

        self.net = []
        for i in range(len(self.hidden_features) - 1):
            hidden = False
            if i < len(self.hidden_features) - 2:
                if nonlinearity is not None:
                    nl = layer_factory(nonlinearity)[0]
                    init = layer_factory(nonlinearity)[1]
                hidden = True
            else:
                if last_nonlinearity is not None:
                    nl = layer_factory(last_nonlinearity)[0]
                    init = layer_factory(last_nonlinearity)[1]

            layer = nn.Linear(self.hidden_features[i], self.hidden_features[i + 1])

            if (hidden and (nonlinearity is not None)) or ((not hidden) and (last_nonlinearity is not None)):
                init(layer)
                self.net.append(layer)
                self.net.append(nl)
                if hidden and dropout > 0:
                    self.net.append(nn.Dropout(p=dropout))
            else:
                if zero_init:
                    layer.weight.data.fill_(1e-4)
                self.net.append(layer)

            if hidden:
                if group_norm > 0:
                    self.net.append(nn.GroupNorm(num_groups=group_norm, num_channels=self.hidden_features[i + 1]))
                if batch_norm:
                    self.net.append(nn.BatchNorm1d(num_features=self.hidden_features[i + 1]))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output