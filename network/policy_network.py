import torch.nn.functional as F

from network.common import *


class PolicyNetwork(nn.Module):
    def __init__(self, num_states, num_actions, layers, activation):
        """Policy Network"""
        super(PolicyNetwork, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions

        self.mlp = MLP([num_states] + layers + [num_actions], activation=activation)
        self.apply(init_weights_zero)

    def forward(self, obs):
        """Forward"""
        action_probs = F.softmax(self.mlp(obs), dim=-1)
        return action_probs

    def reset(self):
        self.apply(init_weights_zero)
