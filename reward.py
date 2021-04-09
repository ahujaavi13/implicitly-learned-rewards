from collections import deque

from network.reward_network import RewardNetwork
from network.gamma_network import GammaNetwork

from utils.common import *


class InternalReward:
    def __init__(self, num_states, num_actions, learning_rate, T3, device):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.device = device
        self.T3 = T3
        self.p_reward = RewardNetwork(num_states=self.num_states, num_actions=self.num_actions, layers=[])
        self.p_gamma = GammaNetwork(val=self.num_states, layers=[]).to(self.device)
        self.optimizer = torch.optim.SGD(self.p_reward.parameters(), lr=self.learning_rate)
        self.gamma_optimizer = torch.optim.SGD(self.p_gamma.parameters(), lr=self.learning_rate)
        self.trajectories = deque(maxlen=self.T3)

    def get_reward_network(self):
        return self.p_reward

    def get_reward_optimizer(self):
        return self.optimizer

    def get_gamma_network(self):
        return self.p_gamma

    def get_one_trajectory(self):
        if len(self.trajectories) > 0:
            return self.trajectories.popleft()
        return None

    def clear_trajectories(self):
        self.trajectories = deque(maxlen=self.T3)

    def learn_reward(self, env, agent, policy, p_reward, p_gamma, max_trajectory_len):

        num_params_policy = sum(p.numel() for p in policy.parameters())
        num_params_reward = sum(p.numel() for p in self.p_reward.parameters())
        num_params_gamma = sum(p.numel() for p in self.p_gamma.parameters())

        assert num_params_policy == num_params_reward, \
            f"Total parameters in policy {num_params_policy} and rewards {num_params_reward} network should be equal"

        c = torch.zeros((1, num_params_policy), dtype=torch.float64)  # 1, num_params_policy
        H = torch.ones((1, num_params_policy), dtype=torch.float64)  # 1, num_params_policy
        A = torch.zeros((1, num_params_reward), dtype=torch.float64)  # 1, num_params_reward
        b = torch.zeros((1, num_params_gamma), dtype=torch.float64)  # 1, num_params_gamma

        for t3 in range(self.T3 + 1):

            states, actions, inner_rewards, outer_rewards, gammas = \
                generate_trajectory(env=env, agent=agent, policy=policy, p_reward=p_reward,
                                    p_gamma=p_gamma, max_trajectory_len=max_trajectory_len)

            log_prob = agent.calc_log_probs(states=states, actions=actions, policy=policy)

            T = len(states)

            # for intermediate calculations
            opt_grads = torch.zeros((T, num_params_policy), dtype=torch.float64)  # T, num_params_policy
            r_phi_grads = torch.zeros((T, num_params_reward), dtype=torch.float64)  # T, num_params_reward
            psi = torch.zeros((T, num_params_policy), dtype=torch.float64)  # T, num_params_policy
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

                psi[t] = torch.sum(opt_grads, dim=0)  # num_params_policy
                for _t in range(t, T):  # TODO: Vectorize
                    cum_p_gammas[t][_t] = torch.prod(gammas[t:_t + 1])

            for t in range(T):
                cum_r_phi_grads[t] = torch.sum(r_phi_grads[t:].T * cum_p_gammas[t][t:], dim=1)
                cum_p_gamma_grads[t] = torch.sum(p_gamma_grads[t:].T * cum_p_gammas[t][t:] * t, dim=1)
                cum_inner_rewards[t] = torch.sum(inner_rewards[t:])
                cum_log_probs[t] = torch.sum(log_prob[t:])

            psi = psi.T
            c += torch.sum(psi * torch.tensor(outer_rewards), dim=1).unsqueeze(0)  # 1, num_params_policy
            H += torch.sum((psi * psi) * inner_rewards, dim=1).unsqueeze(0).detach()  # 1, num_params_policy
            A += torch.sum(opt_grads * cum_r_phi_grads, dim=0).unsqueeze(0)  # 1, num_params_reward
            _b = torch.sum(cum_p_gamma_grads.squeeze() * (cum_inner_rewards - 1. * cum_log_probs), dim=0)
            b += torch.sum(opt_grads * _b.unsqueeze(0).T)

            self.trajectories.append((states, actions, inner_rewards, outer_rewards, gammas, log_prob))

        c = c / self.T3
        H = 1 / (H / self.T3) if torch.sum(H) != 0 else torch.ones_like(H)
        A = A / self.T3
        f_grads = -(c * H * A).float()
        f_gamma_grads = -(torch.mm(c, H.T) * b - self.p_gamma.mlp[0][0].weight).float()
        self.optimizer.zero_grad()
        self.gamma_optimizer.zero_grad()
        self.p_reward.mlp[0][0].weight.grad = f_grads
        self.p_gamma.mlp[0][0].weight.grad = f_gamma_grads
        self.optimizer.step()
        # self.gamma_optimizer.step()

    def eval(self):
        states, actions = [], []
        for s in range(self.num_states):
            for a in range(self.num_actions):
                states.append(one_hot_single_value(s, self.num_states))
                actions.append(torch.tensor(a))

        s_action = one_hot_two_value(cur_val_1=states, cur_val_2=actions,
                                     total_vals_1=self.num_states, total_vals_2=self.num_actions)

        with torch.no_grad():
            inner_rewards = self.p_reward(s_action.float())

        return clip_tensor(inner_rewards)
