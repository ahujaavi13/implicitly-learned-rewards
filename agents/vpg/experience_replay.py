import random

from collections import deque


class ExperienceReplay:
    def __init__(self, maxlen=100000):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=self.maxlen)

    def reset(self):
        self.buffer = deque(maxlen=self.maxlen)

    def sample_uniform(self):
        if self.__len__():
            return random.choice(self.buffer)
        raise Exception(f"No trajectories in memory")

    def add(self, D2):
        self.buffer.extend(D2)

    def __len__(self):
        return len(self.buffer)
