import numpy as np
import torch
from torch.distributions import Categorical

from utils.common import one_hot_single_value
from utils.common import Basis, Fourier_Basis


class VanillaPolicyGradient:
    def __init__(self, num_states, num_actions, env_width, is_state_continuous, args, env, optim):
        self.num_states = num_states
        self.num_actions = num_actions
        self.is_state_continuous = is_state_continuous
        self.env_width = env_width
        self.args = args
        self.env = env
        self.optim = optim

        self.state_features = Fourier_Basis(config=self.args, env=self.env, optim=self.optim)

    def get_action(self, current_state, policy):
        if not self.is_state_continuous:
            state = one_hot_single_value(cur_val=current_state, total_vals=self.num_states, width=self.env_width)
        else:
            state = self.state_features.forward(torch.tensor(current_state).float())

        prob = policy(torch.tensor(state).unsqueeze(0).float())
        sampler = Categorical(prob)
        action = sampler.sample()

        return state, action

    def get_reward_gamma(self, states, actions, p_reward, p_gamma):
        s_action = torch.stack([*states])
        # s_action = torch.Tensor(states)

        a_indices = torch.cat([*actions], dim=-1)
        r_phi = p_reward(s_action.float())
        r_phi = r_phi[torch.arange(r_phi.size(0)), a_indices]

        # Placeholder only
        # gamma = p_gamma(s_action.float())
        # gamma = p_gamma(torch.ones((len(states), 1)).float())
        gamma = p_gamma(torch.ones_like(s_action))

        return r_phi, gamma

    def calc_log_probs(self, states, actions, policy):
        """TODO: Repeated calculation, fix this later"""
        # states = torch.tensor(states).float()
        actions = torch.tensor(actions).float()
        prob = policy(torch.stack([*states], dim=0))
        sampler = Categorical(prob)

        return sampler.log_prob(actions)

    def learn(self, log_prob, rewards, optimizer, gammas):

        T = len(log_prob)

        G = []
        rewards = np.array(rewards)

        for i in range(T):
            G += [np.sum(rewards[i:] * (gammas[i].item() ** np.array(range(i, T))))]

        inner_loss = torch.sum(-log_prob * torch.tensor(G))

        optimizer.zero_grad()
        inner_loss.backward(retain_graph=True)
        optimizer.step()
