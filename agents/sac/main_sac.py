# import pybullet_envs
import gym
import numpy as np
import torch

from sac_torch import Agent
from utils import plot_learning_curve
from gym import wrappers
from envs.SimpleBandit import SimpleBandit
import pickle

from envs.GridWorldSAS import Gridworld_SAS

# torch.manual_seed(42)
# np.random.seed(42)


def one_hot_single_value(cur_val, total_vals):
    """Coverts cur_val into one-hot vector of size total_vals"""
    x = [0] * total_vals
    x[cur_val] = 1
    return x


if __name__ == '__main__':
    env = SimpleBandit(num_states=10, num_actions=4, max_trajectory_len=20)
    n_states, n_actions = env.state_space.n, env.action_space.n

    agent = Agent(input_dims=[env.state_space.n], env=env, n_actions=env.action_space.n)
    n_games = 2000
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    # env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'simple_bandit.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation, _ = env.reset()
        observation = one_hot_single_value(observation, n_states)
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            observation_ = one_hot_single_value(observation_, n_states)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                pass
            observation = observation_
        agent.learn()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

    with open('scores/score_history__.p', 'wb') as fp:
        pickle.dump(score_history, fp)
