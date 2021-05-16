import gym
import matplotlib.pyplot  as plt
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.ticker import NullLocator
import torch
import numpy as np
from gym import error, spaces, utils
from gym.spaces import Discrete, Box


class Gridworld_687(object):
    def __init__(self, max_steps=200, action_prob=1.0, debug=True):
        self.debug = debug
        self.action_prob = action_prob
        self.randomness = 1 - action_prob
        self.n_actions = 4

        self.n_observations = 25 #TODO CHECK THIS
        self.width = np.sqrt(self.n_observations).astype(np.int32)

        self.action_space = Discrete(self.n_actions)
        self.observation_space = Box(low=np.zeros(2, dtype=np.int32), high=np.full(2, self.width, dtype=np.int32), dtype=np.int32)
        self._n_episodes = 0

        self.max_steps = max_steps
        self.step_reward = 0
        self.step_unit = 1
        self.repeat = 1

        self.disp_flag = False

        self.collision_reward = 0
        self.movement_reward = 0

        self.static_obstacles = self.get_static_obstacles()


        if debug:
            self.heatmap = np.zeros((self.width, self.width))

        self.reset()

    def seed(self, seed):
        self.seed = seed

    def get_embeddings(self):
        pass # TODO

# TODO fix dims
    def render(self):
        x, y = self.curr_pos

        if not self.disp_flag:
            self.disp_flag = True
        
            self.currentAxis = plt.gca()
            plt.figure(1, frameon=False) #Turns off the the boundary padding
            # self.currentAxis.xaxis.set_major_locator(NullLocator()) #Turns of ticks of x axis
            # self.currentAxis.yaxis.set_major_locator(NullLocator()) #Turns of ticks of y axis
            plt.ion()                                               #To avoid display blockage
            
            self.circle = Circle((x / (self.width-1), 1 - y / (self.width-1)), 0.1, color='red')
            for coords in self.static_obstacles:
                x1, y1, _, _ = coords
                self.currentAxis.add_patch(Rectangle((x1 / (self.width-1), 1 - y1 / (self.width-1)), 1/self.width, 1/self.width, fill=True, color='gray'))

        # for key, val in self.dynamic_obs.items():
        #     pass

        for key, val in self.reward_states.items():
            coords, cond = val
            if cond:
                x1, y1, _, _ = coords
                self.objects[key] = Rectangle((x1 / (self.width-1), 1 - y1/ (self.width-1)), 1/self.width, 1/self.width, fill=True)
                self.currentAxis.add_patch(self.objects[key])
                
        # Arrow
        
        self.objects['circle'] = Circle((x / (self.width-1), 1 - y / (self.width-1)), 0.01, color='red')
        self.currentAxis.add_patch(self.objects['circle'])

        plt.pause(1e-1)
        for _, item in self.objects.items():
            item.remove()
        self.objects = {}
    
    def get_env_state(self):
        return self.curr_state.copy(), 0, self.is_terminal()

    # reward values associated with states
    def set_rewards(self):
        # All rewards
        self.G1_reward = -10 #100
        self.G2_reward = +10 #- 5

    def reset(self):
        self.set_rewards()
        self.steps_taken = 0
        self.reward_states = self.get_reward_states()
        self.dynamic_obs = self.get_dynamic_obstacles()
        self.objects = {}

        # self.curr_pos = torch.Tensor([0, 0], dtype=torch.int32)
        self.curr_pos = np.array([0, 0], dtype=np.int32)

        self.action_state_offset = {
            -1: np.array([0,0]),
            0: np.array([0,-1]),
            1: np.array([1,0]),
            2: np.array([0,1]),
            3: np.array([-1,0])
        }
        self.curr_state = self.make_state()

        self._n_episodes += 1
        return self.curr_state, self.get_valid_actions()

    def get_valid_actions(self):
        self.valid_actions = np.array((np.random.rand(self.n_actions) <= self.action_prob), dtype=int)
        # Make sure that there is at least one available action always.
        while not self.valid_actions.any():
            self.valid_actions = np.array((np.random.rand(self.n_actions) <= self.action_prob), dtype=int)

        return self.valid_actions

    def step(self, action):
        assert self.valid_actions[action]

        # print(self.steps_taken)
        self.steps_taken += 1
        reward = 0

        term = self.is_terminal()
        # print(term)
        if term:
            return self.curr_state, 0, term, None

        reward += self.step_reward

        actual_action = action
        transition = True
        # Add Randomness
        rng = np.random.rand()
        if rng < self.randomness:
            if rng < self.randomness / 4:
                actual_action = (action - 1) % self.n_actions
            elif rng < self.randomness / 2:
                actual_action = (action + 1) % self.n_actions
            else:
                actual_action = -1

        new_pos = self.curr_pos + self.action_state_offset[actual_action]

        if self.valid_pos(new_pos):
            reward += self.movement_reward
            self.curr_pos = new_pos
            reward += self.get_goal_rewards(self.curr_pos)
        else:
            reward += self.collision_reward
            transition = False

        if self.is_terminal():
            transition = False
        
        if transition:
            self.curr_state = self.make_state()

        if self.debug:
            # print(self.curr_pos)
            pass

        return self.curr_state.copy(), reward, self.is_terminal(), None

    def make_state(self):
        x, y = self.curr_pos
        state = [x, y]

        return state

    def get_goal_rewards(self, pos):
        for key, val in self.reward_states.items():
            region, reward = val
            if reward and self.in_region(pos, region):
                self.reward_states[key]
                if self.debug: print("Got reward {} in {} steps!! ".format(reward, self.steps_taken))
                
                return reward
        return 0

    # States where reward exist
    def get_reward_states(self):
        self.G1 = (2,4,2,4)
        self.G2 = (4,4,4,4)
        return {"G1": (self.G1, self.G1_reward),
                "G2": (self.G2, self.G2_reward)}

    # returns the list of obstacles
    def get_static_obstacles(self):
        self.O1 = (2,2,2,2)
        self.O2 = (2,3,2,3)
        obstacles = [self.O1, self.O2]
        
        return obstacles

    def get_dynamic_obstacles(self):
        return []

    # Returns whether a position is valid
    def valid_pos(self, pos):
        flag = True

        # Check position within boundaries
        if not self.in_region(pos, [0,0,self.width - 1,self.width - 1]):
            # print(pos, "not valid")
            flag = False

        # Checks if position is in obstacle
        for region in self.static_obstacles:
            if self.in_region(pos, region):
                flag = False
                break

        return flag

    # Pos: (x,y)
    # Region ((x1,y1), (x2,y2))
    def in_region(self, pos, region):
        x0, y0 = pos
        x1, y1, x2, y2 = region
        if x0 >= x1 and x0 <= x2 and y0 >= y1 and y0 <= y2:
            return True
        else:
            return False

    def is_terminal(self):
        if self.in_region(self.curr_pos, self.G2):
            return True
        elif self.steps_taken >= self.max_steps:
            return True
        else:
            return False

if __name__=="__main__":
    # Random Agent
    np.random.seed(0)
    rewards_list = []
    env = Gridworld_687(debug=True)
    for i in range(50):
        rewards = 0
        done = False
        _, valid = env.reset()
        while not done:
            # env.render()
            # plt.show()
            available = np.where(valid)[0]
            action = np.random.choice(available)
            next_state, r, valid, done, _ = env.step(action)
            rewards += r

        # print(env.steps_taken)
        rewards_list.append(rewards)

    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))