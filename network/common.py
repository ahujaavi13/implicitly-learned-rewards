import torch
import torch.nn as nn


def MLP(layers, activation=True, bias=False):
    """Returns MLP block of depth = len(layers)-1. If activation is false, "layers" is ignored
        and only a linear layer of size (layer[0], layer[-1]) is returned."""

    if activation:
        return nn.Sequential(
            *[nn.Sequential(nn.Linear(layers[i - 1], layers[i], bias=bias),
                            nn.LeakyReLU(negative_slope=-10.0)) for i in range(1, len(layers))]
        )
    else:
        return nn.Sequential(
            *[nn.Sequential(nn.Linear(layers[0], layers[-1], bias=bias))]
        )


def init_weights_zero(module):
    """Initializes all linear layers with weight zeros."""
    if isinstance(module, nn.Linear):
        module.weight.data.fill_(0.)
        if module.bias is not None:
            module.bias.data.zero_()


def init_weights_gaussian(module):
    """Initializes all linear layers with guassian(0., .02)"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0., std=.02)
        if module.bias is not None:
            module.bias.data.zero_()


def init_weights_gamma(module):
    """Initializes Gamma with 0.9"""
    if isinstance(module, nn.Linear):
        module.weight.data.fill_(0.9)
        if module.bias is not None:
            module.bias.data.zero_()


def init_weights_saved(module):
    """Initializes reward with weight pre-trained weight"""
    if isinstance(module, nn.Linear):
        weights = torch.load('./checkpoint/2/reward99.pth')['mlp.0.0.weight']
        module.weight.data = weights
        if module.bias is not None:
            module.bias.data.zero_()
