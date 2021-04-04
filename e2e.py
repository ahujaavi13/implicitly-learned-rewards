import torch
import numpy as np
from collections import deque

from network.policy_network import PolicyNetwork
from agents.vpg import VanillaPolicyGradient
from reward import InternalReward

from envs.SimpleBandit import SimpleBandit
from utils.common import *

import matplotlib.pyplot as plt

# Hyperparameters
T1, T2 = 100, 2000
max_trajectory_len, T3 = 500, 100
policy_lr, reward_lr = 1e-2, 1e-4
seed = 42

torch.manual_seed(seed)

inner_returns, outer_returns = deque(maxlen=100), deque(maxlen=100)
plot_outer_returns = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = SimpleBandit(num_states=10, num_actions=4, max_trajectory_len=max_trajectory_len)

internal_reward = InternalReward(num_states=env.state_space.n, num_actions=env.action_space.n,
                                 learning_rate=reward_lr, T3=T3, device=device)

policy = PolicyNetwork(num_states=env.state_space.n, num_actions=env.action_space.n,
                       layers=[], activation=False)

optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
agent = VanillaPolicyGradient(num_states=env.state_space.n, num_actions=env.action_space.n)

p_reward = internal_reward.get_reward_network()
p_gamma = internal_reward.get_gamma_network()

for t1 in range(T1):
    policy.reset()
    for t2 in range(T2):
        states, actions, inner_rewards, outer_rewards, gammas = \
            generate_trajectory(env=env, agent=agent, policy=policy, p_reward=p_reward,
                                p_gamma=p_gamma, max_trajectory_len=max_trajectory_len)

        log_prob = agent.calc_log_probs(states=states, actions=actions, policy=policy)
        agent.learn(log_prob=log_prob, rewards=inner_rewards.detach(), optimizer=optimizer, gammas=gammas)

        outer_returns.append(np.sum(outer_rewards))
        inner_returns.append(np.sum(inner_rewards.clone().detach().numpy()))

        if not t2 % 100:
            print(f"Episode: {t2}\t"
                  f"Avg. Outer Return: {np.mean(outer_returns)}\t"
                  f"Avg. Inner Return: {np.mean(inner_returns)}\t")

    internal_reward.learn_reward(env=env, agent=agent, policy=policy, p_reward=p_reward, p_gamma=p_gamma,
                                 max_trajectory_len=max_trajectory_len)

    create_dir_and_save(model=p_reward, parent_dir='checkpoint/3/', filename=f'reward{t1}.pth')

    inner_reward_eval = internal_reward.eval()
    plot_outer_returns.append(list(outer_returns))
    print(f"[{t1}]\tRewards: {inner_reward_eval.reshape(env.action_space.n, env.state_space.n)}")

plt.plot(np.arange(0, T1), np.mean(plot_outer_returns, axis=1))
plt.xlabel("T1")
plt.ylabel("Mean Outer Return")
plt.show()
