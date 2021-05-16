import torch.nn.functional as F

from network.common import *


class RewardNetwork(nn.Module):
    def __init__(self, num_states, num_actions, layers, activation):
        """Reward Network"""
        super(RewardNetwork, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions

        # self.mlp = MLP([num_states * num_actions] + layers + [1], activation=False)
        self.mlp = MLP([num_states] + layers + [num_actions], activation=activation)
        self.reset()

    def forward(self, s_actions):
        """Forward"""
        return F.softmax(self.mlp(s_actions), dim=-1)

    def reset(self):
        self.apply(init_weights_zero)
