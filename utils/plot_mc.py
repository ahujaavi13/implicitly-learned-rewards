import numpy as np
import torch
import matplotlib.pyplot as plt


from envs.MountainCar import MountainCarEnv
from utils.common import _generate_trajectory
from agents.vpg.vpg_basis import VanillaPolicyGradient
from network.policy_network import PolicyNetwork
from network.reward_network import RewardNetwork
from network.gamma_network import GammaNetwork


class args:
    def __init__(self):
        self.fourier_order = 3
        self.state_lr = 1e-3
        self.fourier_coupled = True
        self.device = 'cpu'


def plot_mc(x, v, T, it, cn):

    states = []
    for s in x:
        for vel in v:
            states.append([s, vel])

    print(torch.tensor(states).size())

    env = MountainCarEnv()
    n_states, n_actions, env_width = env.observation_space.shape[0], env.action_space.n, None
    is_state_continuous = True
    a = args()

    agent = VanillaPolicyGradient(num_states=n_states, num_actions=n_actions, env_width=env_width,
                                  is_state_continuous=is_state_continuous, args=a, env=env, optim=torch.optim.Adam)

    layers = [64, 64]
    num_states, num_actions = 16, 3
    policy = PolicyNetwork(num_states=16, num_actions=n_actions, layers=layers, activation=True)
    p_reward = RewardNetwork(num_states=num_states, num_actions=num_actions, layers=layers, activation=True)
    p_gamma = GammaNetwork(num_states=num_states, num_actions=num_actions, layers=layers, activation=True)

    policy.load_state_dict(torch.load(f'weights/checkpoint{cn}/policy/policy{it}.pth'))
    p_reward.load_state_dict(torch.load(f'weights/checkpoint{cn}/reward/reward{it}.pth'))
    p_gamma.load_state_dict(torch.load(f'weights/checkpoint{cn}/gamma/gamma{it}.pth'))
    policy.eval()
    p_reward.eval()
    p_gamma.eval()

    _, _, inner_rewards, outer_rewards, gammas = _generate_trajectory(env, agent, policy, p_reward, p_gamma, states)

    # outer_rewards = torch.tensor(outer_rewards)[1:]
    inner_rewards = inner_rewards.detach().numpy()
    gammas = gammas.detach().numpy()

    X, Y = np.meshgrid(x, v)
    R = inner_rewards.reshape(T, T)
    # G = gammas.reshape(T, T)
    # _, R = np.meshgrid(x, inner_rewards)
    # R = np.repeat(inner_rewards[np.newaxis, :], 200, axis=0)
    # print(R[0].all() == R[1].all(), R.shape)
    # exit()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # start = 0
    # end = -1
    # ax.scatter(x[start:end], v[0:-1], inner_rewards[0:-1])
    ax.set_title('Mountain Car - Reward')
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.set_zlabel('rewards')
    plt.savefig(f'plots/{cn}/{it}.png')
    plt.close()
    # plt.show()


if __name__ == "__main__":
    T = 100
    iters = 24
    check_num = 17
    x, v = None, None
    for it in range(iters):
        x, v = np.linspace(-1.2, 0.6, T), np.linspace(-0.07, 0.07, T)
        plot_mc(x, v, T, it, check_num)
    plot_mc(x, v, T, 'init', check_num)
