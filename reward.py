from collections import deque

import torch

from network.reward_network import RewardNetwork
from network.gamma_network import GammaNetwork

from utils.common import *
import random


class InternalReward:
    def __init__(self, num_states, num_actions, env_width, learning_rate, T3, layers, activation, freeze_gamma, device):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.device = device
        self.T3 = T3
        self.p_reward = RewardNetwork(num_states=self.num_states, num_actions=self.num_actions,
                                      layers=layers, activation=activation)
        self.p_gamma = GammaNetwork(val=self.num_states, layers=[])
        self.optimizer = torch.optim.Adam(self.p_reward.parameters(), lr=self.learning_rate)
        self.gamma_optimizer = torch.optim.SGD(self.p_gamma.parameters(), lr=self.learning_rate)
        self.D = deque(maxlen=100000)
        self.freeze_gamma = freeze_gamma
        self.env_width = env_width

    def get_reward_network(self):
        return self.p_reward

    def get_reward_optimizer(self):
        return self.optimizer

    def get_gamma_network(self):
        return self.p_gamma

    def get_one_trajectory_uniform(self):
        return random.choice(self.D)

    def get_one_trajectory(self):
        if len(self.D) > 0:
            return self.D.popleft()
        return None

    def clear_trajectories(self):
        self.D = deque(maxlen=100000)

    def learn_reward(self, env, agent, policy, p_reward, p_gamma):

        num_params_policy = sum(p.numel() for p in policy.parameters())
        num_params_reward = sum(p.numel() for p in self.p_reward.parameters())
        num_params_gamma = sum(p.numel() for p in self.p_gamma.parameters())

        assert num_params_policy == num_params_reward, \
            f"Total parameters in policy {num_params_policy} and rewards {num_params_reward} network should be equal"

        c = torch.zeros((1, num_params_policy), dtype=torch.float64)  # 1, num_params_policy
        H = torch.ones((1, num_params_policy), dtype=torch.float64)  # 1, num_params_policy
        A = torch.zeros((1, num_params_policy), dtype=torch.float64)  # 1, num_params_policy
        b = torch.zeros((1, num_params_gamma), dtype=torch.float64)  # 1, num_params_gamma

        for t3 in range(self.T3):

            states, actions, inner_rewards, outer_rewards, gammas = \
                generate_trajectory(env=env, agent=agent, policy=policy, p_reward=p_reward, p_gamma=p_gamma)

            log_prob = agent.calc_log_probs(states=states, actions=actions, policy=policy)

            T = len(states)

            # for intermediate calculations
            opt_grads = torch.zeros((T, num_params_policy), dtype=torch.float64)  # T, num_params_policy
            r_phi_grads = torch.zeros((T, num_params_reward), dtype=torch.float64)  # T, num_params_reward
            # psi = torch.zeros((T, num_params_policy), dtype=torch.float64)  # T, num_params_policy
            cum_r_phi_grads = torch.zeros((T, num_params_reward), dtype=torch.float64)  # T, num_params_reward
            cum_inner_rewards = torch.zeros((T,), dtype=torch.float64)  # T,
            cum_log_probs = torch.zeros((T,), dtype=torch.float64)  # T,
            cum_p_gammas = torch.zeros((T, T), dtype=torch.float64)  # T, T
            p_gamma_grads = torch.zeros((T, num_params_gamma), dtype=torch.float64)  # T, num_params_gamma
            cum_p_gamma_grads = torch.zeros((T, num_params_gamma), dtype=torch.float64)  # T, num_params_gamma

            for t in range(T):
                inputs = (policy, self.p_reward, self.p_gamma)
                outputs = (log_prob[t], inner_rewards[t], gammas[t])
                retain_graphs = (True, True, True)

                # Calculate gradients of outputs w.r.t inputs
                opt_grads[t], r_phi_grads[t], p_gamma_grads[t] = calc_grads(outputs, inputs, retain_graphs)

                # psi[t] = torch.sum(opt_grads, dim=0)  # num_params_policy
                for _t in range(t, T):  # TODO: Vectorize
                    cum_p_gammas[t][_t] = torch.prod(gammas[t:_t + 1])
                print(cum_p_gammas)
                1/0

            for t in range(T):
                cum_r_phi_grads[t] = torch.sum(r_phi_grads[t:].T * cum_p_gammas[t][t:], dim=1)
                cum_p_gamma_grads[t] = torch.sum(p_gamma_grads[t:].T * cum_p_gammas[t][t:] * t, dim=1)
                cum_inner_rewards[t] = torch.sum(inner_rewards[t:])
                cum_log_probs[t] = torch.sum(log_prob[t:])

            # psi = psi.T
            # Fixed c
            c += torch.sum(psi * torch.tensor(outer_rewards), dim=1).unsqueeze(0)  # 1, num_params_policy
            H += torch.sum((psi * psi) * inner_rewards, dim=1).unsqueeze(0).detach()  # 1, num_params_policy
            A += torch.sum(opt_grads * cum_r_phi_grads, dim=0).unsqueeze(0)  # 1, num_params_reward
            if not self.freeze_gamma:
                _b = torch.sum(cum_p_gamma_grads.squeeze() * (cum_inner_rewards - 1. * cum_log_probs), dim=0)
                b += torch.sum(opt_grads * _b.unsqueeze(0).T)

            self.D.append((states, actions, inner_rewards, outer_rewards, gammas, log_prob))

        c = c / self.T3
        H = 1 / (H / self.T3) if torch.sum(H) != 0 else torch.ones_like(H)
        A = A / self.T3
        self.optimizer.zero_grad()
        self._update_weights(c, H, A, b)
        self.optimizer.step()
        if not self.freeze_gamma:
            self.gamma_optimizer.zero_grad()
            self.gamma_optimizer.step()

    def _update_weights(self, c, H, A, b):
        f_grads = -(c * H * A).float().detach()
        for i, p in enumerate(self.p_reward.parameters()):
            x, y = self.p_reward.mlp[i][0].weight.shape
            _f_grads = f_grads[:, :x * y].view(x, y)
            self.p_reward.mlp[i][0].weight.grad = _f_grads
            f_grads = f_grads[:, x * y:]

            if not self.freeze_gamma:
                raise Exception("Gamma to be implemented")
                # f_gamma_grads = -(torch.mm(c, H.T) * b - self.p_gamma.mlp[i][0].weight).float()
                # self.p_gamma.mlp[0][0].weight.grad = f_gamma_grads

    def eval(self):
        states, actions = [], []
        for s in range(self.num_states):
            for a in range(self.num_actions):
                states.append(one_hot_single_value(cur_val=s, total_vals=self.num_states, width=self.env_width))
                actions.append(torch.tensor(a))

        with torch.no_grad():
            # inner_rewards = self.p_reward(s_action.float())
            inner_rewards = self.p_reward(torch.tensor(states).float())
            a_indices = torch.stack(actions)
            inner_rewards = inner_rewards[torch.arange(inner_rewards.size(0)), a_indices]

        return clip_tensor(inner_rewards)
