import numpy as np
import torch
from torch.distributions import Categorical

from utils.common import one_hot_single_value, one_hot_two_value


class VanillaPolicyGradient:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

    def get_action(self, current_state, policy):
        state = one_hot_single_value(cur_val=current_state, total_vals=self.num_states)
        prob = policy(torch.tensor(state).unsqueeze(0).float())
        sampler = Categorical(prob)
        action = sampler.sample()

        return state, action

    def get_reward_gamma(self, states, actions, p_reward, p_gamma):
        s_action = one_hot_two_value(cur_val_1=states, cur_val_2=actions,
                                     total_vals_1=self.num_states, total_vals_2=self.num_actions)
        r_phi = p_reward(s_action.float())

        # Placeholder only
        gamma = p_gamma(torch.ones_like(torch.tensor(states).float()))
        # gamma = p_gamma(torch.ones((len(states), 1)).float())

        return r_phi, gamma

    def calc_log_probs(self, states, actions, policy):
        """TODO: Repeated calculation, fix this later"""
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).float()
        prob = policy(states)
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
