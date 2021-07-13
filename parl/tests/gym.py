# mock gym environment
import numpy as np
from random import random


def make(env_name):
    print('>>>>>>>>> you are testing mock gym env: ', env_name)
    if env_name == 'CartPole-v0':
        return CartPoleEnv()


class CartPoleEnv(object):
    def __init__(self):
        class ActionSpace(object):
            def __init__(self):
                self.n = 2

        class ObservationSpace(object):
            def __init__(self):
                self.shape = (4, )

        self.observation_space = ObservationSpace()
        self.action_space = ActionSpace()

    def step(self, action):
        action = int(action)
        obs = np.random.random(4) * 2 - 1
        reward = random() * 2 - 1
        done = True if random() < 0.05 else False
        info = {}
        return obs, reward, done, info

    def reset(self):
        obs = np.random.random(4) * 2 - 1
        return obs
