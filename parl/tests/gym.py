#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# mock gym environment
import numpy as np
from random import random


def make(env_name):
    print('>>>>>>>>> you are testing mock gym env: ', env_name)
    if env_name == 'CartPole-v0':
        return CartPoleEnv()
    elif env_name == 'PongNoFrameskip-v4':
        return PongEnv()
    elif env_name == 'HalfCheetah-v1':
        return HalfCheetahEnv()
    else:
        raise NotImplementedError(
            'Mock env not defined, please check your env name')


# mock Box
class Box(object):
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


# mock gym.Wrapper
class Wrapper(object):
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)


# mock gym.ObservationWrapper
class ObservationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info


# mock gym.RewardWrapper
class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)


# Atari Specific
# mock env.action_space
class ActionSpace(object):
    def __init__(self, n, shape=None):
        self.n = n
        self.shape = shape


# mock env.observation_space
class ObservationSpace(object):
    def __init__(self, dim, dtype):
        self.shape = dim
        self.dtype = dtype


# mock env.spec
class Spec(object):
    def __init__(self, id='PongNoFrameskip-v4'):
        self.id = id


# mock gym.spaces
class spaces(object):
    def __init__(self):
        pass

    @staticmethod
    def Box(high, low, shape, dtype):
        return ObservationSpace(shape, dtype)


# mock CartPole-v0
class CartPoleEnv(object):
    def __init__(self):
        self.observation_space = ObservationSpace((4, ), dtype='int8')
        self.action_space = ActionSpace(2)

    def step(self, action):
        action = int(action)
        obs = np.random.random(4) * 2 - 1
        reward = np.random.choice([0.0, 1.0])
        done = np.random.choice([True, False], p=[0.1, 0.9])
        info = {}
        return obs, reward, done, info

    def reset(self):
        obs = np.random.random(4) * 2 - 1
        return obs

    def seed(self, val):
        pass

    def close(self):
        pass


# mock PongNoFrameskip-v4
class PongEnv(object):
    def __init__(self):
        class Lives(object):
            def lives(self):
                return np.random.randint(0, 5)

        class Ale(object):
            def __init__(self):
                self.ale = Lives()
                self.np_random = np.random

            def get_action_meanings(self):
                return ['NOOP'] * 6

        self.observation_space = ObservationSpace((210, 160, 3), 'unit8')
        self.action_space = ActionSpace(6)
        self.unwrapped = Ale()
        self.metadata = {'render.modes': []}
        self.reward_range = [0, 1]
        self.spec = Spec('PongNoFrameskip-v4')

    def step(self, action):
        action = int(action)
        obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        reward = np.random.choice([0.0, 1.0])
        done = np.random.choice([True, False], p=[0.1, 0.9])
        info = {}
        return obs, reward, done, info

    def reset(self):
        obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        return obs

    def close(self):
        pass

    def seed(self, val):
        pass


# mock mujoco envs
class HalfCheetahEnv(object):
    def __init__(self):
        self.observation_space = Box(
            high=np.array([np.inf] * 17),
            low=np.array([-np.inf] * 17),
            shape=(17, ),
            dtype=None)
        self.action_space = Box(
            high=np.array([1.0] * 6),
            low=np.array([-1.0] * 6),
            shape=(6, ),
            dtype=None)
        self._max_episode_steps = 1000
        self._elapsed_steps = 0

    def step(self, action):
        obs = np.random.randn(17)
        reward = np.random.choice([0.0, 1.0])
        done = np.random.choice([True, False], p=[0.01, 0.99])
        info = {}
        return obs, reward, done, info

    def reset(self):
        obs = np.random.randn(17)
        return obs

    def seed(self, val):
        pass

    def close(self):
        pass
