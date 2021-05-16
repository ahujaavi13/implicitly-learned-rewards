import numpy as np
import hydra
from collections import deque
from envs.SimpleBandit import SimpleBandit
from envs.GridWorldSAS import Gridworld_SAS
from envs.Gridworld_687 import Gridworld_687
from network.policy_network import PolicyNetwork
from agents.vpg.vpg import VanillaPolicyGradient
from reward import InternalReward
from utils.common import *
from utils.plots import plot_off_policy_on_policy_comparison


def train(args):
    reuse_trajectories = args.reuse_trajectories
    env_name = args.env
    torch.manual_seed(args.seed)
    layers = args.layers.split(',') if args.layers != 'None' else []
    layers = list(map(int, filter(None, layers)))

    # For stats/graphs only
    inner_returns, outer_returns = deque(maxlen=100), deque(maxlen=100)
    plot_inner_returns, plot_outer_returns = [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if env_name == "SimpleBandit":
        env = SimpleBandit(num_states=10, num_actions=4, max_trajectory_len=args.max_trajectory_len)
        n_states, n_actions, env_width = env.state_space.n, env.action_space.n, None
        is_state_continuous = False
    elif env_name == "GridWorldSAS":
        env = Gridworld_SAS(debug=True)
        n_states, n_actions, env_width = env.observation_space.shape[0], env.action_space.shape[0], None
        is_state_continuous = True
    elif env_name == "GridWorld":  # TODO - Added Matthew Gridworld
        env = Gridworld_687()
        n_states, n_actions, env_width = env.n_observations, env.action_space.n, env.width
        is_state_continuous = False
    else:
        raise Exception("Environment not found")

    internal_reward = InternalReward(num_states=n_states, num_actions=n_actions, env_width=env_width, learning_rate=args.reward_lr,
                                     T3=args.T3, layers=layers, activation=args.activation,
                                     freeze_gamma=args.freeze_gamma, device=device)

    policy = PolicyNetwork(num_states=n_states, num_actions=n_actions, layers=layers, activation=args.activation)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    agent = VanillaPolicyGradient(num_states=n_states, num_actions=n_actions, env_width=env_width, is_state_continuous=is_state_continuous)

    p_reward = internal_reward.get_reward_network()
    p_gamma = internal_reward.get_gamma_network()

    for t1 in range(args.T1):
        for t2 in range(args.T2):
            if not reuse_trajectories or not t1:
                states, actions, inner_rewards, outer_rewards, gammas = \
                    generate_trajectory(env=env, agent=agent, policy=policy, p_reward=p_reward, p_gamma=p_gamma)

                log_prob = agent.calc_log_probs(states=states, actions=actions, policy=policy)
            else:
                states, actions, inner_rewards, outer_rewards, gammas, log_prob = \
                    internal_reward.get_one_trajectory_uniform()
                
                # NOTE MATT
                log_prob = agent.calc_log_probs(states, actions, policy)

            agent.learn(log_prob=log_prob, rewards=inner_rewards.detach(), optimizer=optimizer, gammas=gammas.detach())
            outer_returns.append(np.sum(outer_rewards))
            inner_returns.append(np.sum(inner_rewards.clone().detach().numpy()))

            if not (t2+1) % 100:
                print(f"Episode: {t2}\t"
                      f"Avg. Outer Return: {np.mean(outer_returns)}\t"
                      f"Avg. Inner Return: {np.mean(inner_returns)}\t")

        internal_reward.learn_reward(env=env, agent=agent, policy=policy, p_reward=p_reward, p_gamma=p_gamma)

        create_dir_and_save(model=p_reward, parent_dir='checkpoint/4/', filename=f'reward{t1}.pth')

        if not reuse_trajectories:
            plot_outer_returns.append(list(outer_returns))
        else:
            plot_outer_returns.append(np.mean(outer_returns))
        plot_inner_returns.append(list(inner_returns))

        # inner_reward_eval = internal_reward.eval()
        # print(f"[{t1}]\tRewards: {inner_reward_eval.reshape(n_states, n_actions)}")

    return plot_outer_returns


@hydra.main(config_path='config', config_name='config')
def main(args):
    print(args.pretty())
    plot_outer_returns_off_policy = train(args=args)
    args.reuse_trajectories = False
    plot_outer_returns_on_policy = train(args=args)
    plot_off_policy_on_policy_comparison(T1=args.T1, T2=args.T2, T3=args.T3,
                                         on_policy_vals=plot_outer_returns_on_policy,
                                         off_policy_vals=plot_outer_returns_off_policy)


if __name__ == "__main__":
    main()
