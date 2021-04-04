from network.common import *


class GammaNetwork(nn.Module):
    def __init__(self, val, layers):
        """Gamma Network"""
        super(GammaNetwork, self).__init__()

        self.val = val
        self.mlp = MLP([self.val] + layers + [1])
        self.apply(init_weights_gamma)

    def forward(self, obs):
        """Forward"""
        return self.mlp(obs)

    def reset(self):
        self.apply(init_weights_gamma)
