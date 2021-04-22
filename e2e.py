import numpy as np
import argparse
from collections import deque

from network.policy_network import PolicyNetwork
from agents.vpg import VanillaPolicyGradient
from reward import InternalReward
from envs.SimpleBandit import SimpleBandit
from envs.GridWorldSAS import Gridworld_SAS
from utils.common import *
from utils.plots import plot_off_policy_on_policy_comparison


def train(args):
    reuse_trajectories = args['reuse_trajectories']

    torch.manual_seed(seed)

    # For stats/graphs only
    inner_returns, outer_returns = deque(maxlen=100), deque(maxlen=100)
    plot_inner_returns, plot_outer_returns = [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = SimpleBandit(num_states=10, num_actions=4, max_trajectory_len=max_trajectory_len)
    n_states, n_actions = env.state_space.n, env.action_space.n

    # env = Gridworld_SAS(debug=True)
    # n_states, n_actions = env.observation_space.shape[0], env.action_space.shape[0]

    is_state_continuous = False

    internal_reward = InternalReward(num_states=n_states, num_actions=n_actions,
                                     learning_rate=reward_lr, T3=T3, device=device)

    policy = PolicyNetwork(num_states=n_states, num_actions=n_actions,
                           layers=[], activation=False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    agent = VanillaPolicyGradient(num_states=n_states, num_actions=n_actions, is_state_continuous=is_state_continuous)

    p_reward = internal_reward.get_reward_network()
    p_gamma = internal_reward.get_gamma_network()

    for t1 in range(T1):
        for t2 in range(T2):
            if not reuse_trajectories or not t1:
                states, actions, inner_rewards, outer_rewards, gammas = \
                    generate_trajectory(env=env, agent=agent, policy=policy, p_reward=p_reward,
                                        p_gamma=p_gamma, max_trajectory_len=max_trajectory_len)

                log_prob = agent.calc_log_probs(states=states, actions=actions, policy=policy)
            else:
                states, actions, inner_rewards, outer_rewards, gammas, log_prob = \
                    internal_reward.get_one_trajectory_uniform()

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
        if not reuse_trajectories:
            plot_outer_returns.append(list(outer_returns))
        else:
            plot_outer_returns.append(np.mean(outer_returns))
        plot_inner_returns.append(list(inner_returns))

        print(f"[{t1}]\tRewards: {inner_reward_eval.reshape(n_states, n_actions)}")

    return plot_outer_returns


if __name__ == "__main__":
    # Hyperparameters
    T1, T2, T3 = 100, 100, 40
    max_trajectory_len = 20
    policy_lr, reward_lr = 1e-3, 1e-4
    seed = 42

    parser = argparse.ArgumentParser(description="Train Reward")
    parser.add_argument("--reuse_trajectories", action="store_true",
                        help="Trains the policy using trajectories generated for learning reward")

    p_args = vars(parser.parse_args())
    plot_outer_returns_off_policy = train(args=p_args)
    p_args['reuse_trajectories'] = False
    plot_outer_returns_on_policy = train(args=p_args)
    plot_off_policy_on_policy_comparison(T1=T1, T2=T2, T3=T3,
                                         on_policy_vals=plot_outer_returns_on_policy,
                                         off_policy_vals=plot_outer_returns_off_policy)
