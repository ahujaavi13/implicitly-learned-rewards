from network.common import *


class RewardNetwork(nn.Module):
    def __init__(self, num_states, num_actions, layers):
        """Reward Network"""
        super(RewardNetwork, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions

        # self.mlp = MLP([num_states * num_actions] + layers + [1], activation=False)
        self.mlp = MLP([num_states] + layers + [num_actions], activation=False)
        self.apply(init_weights_zero)

    def forward(self, s_actions):
        """Forward"""
        return self.mlp(s_actions)

    def reset(self):
        self.apply(init_weights_zero)
