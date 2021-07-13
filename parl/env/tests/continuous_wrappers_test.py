#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import gym
import numpy as np
import unittest
from parl.env.continuous_wrappers import ActionMappingWrapper


class MockEnv(gym.Env):
    def __init__(self, low, high):
        self.action_space = gym.spaces.Box(
            low=np.array(low), high=np.array(high))
        self._max_episode_steps = 1000

    def step(self, action):
        self.action = action

    def reset(self):
        return None


class TestActionMappingWrapper(unittest.TestCase):
    def test_action_mapping(self):
        origin_act = np.array([-1.0, 0.0, 1.0])

        env = MockEnv([0.0] * 3, [1.0] * 3)
        wrapper_env = ActionMappingWrapper(env)
        wrapper_env.step(origin_act)
        self.assertListEqual(list(env.action), [0.0, 0.5, 1.0])

        env = MockEnv([-2.0] * 3, [2.0] * 3)
        wrapper_env = ActionMappingWrapper(env)
        wrapper_env.step(origin_act)
        self.assertListEqual(list(env.action), [-2.0, 0.0, 2.0])

        env = MockEnv([-5.0] * 3, [10.0] * 3)
        wrapper_env = ActionMappingWrapper(env)
        wrapper_env.step(origin_act)
        self.assertListEqual(list(env.action), [-5.0, 2.5, 10.0])

        # test low bound or high bound is different in different dimensions.
        env = MockEnv([0.0, -2.0, -5.0], [1.0, 2.0, 10.0])
        wrapper_env = ActionMappingWrapper(env)
        wrapper_env.step(origin_act)
        self.assertListEqual(list(env.action), [0.0, 0.0, 10.0])

        origin_act = np.array([0.0, 0.0, 0.0])
        env = MockEnv([0.0, -2.0, -5.0], [1.0, 2.0, 10.0])
        wrapper_env = ActionMappingWrapper(env)
        wrapper_env.step(origin_act)
        self.assertListEqual(list(env.action), [0.5, 0.0, 2.5])


if __name__ == '__main__':
    unittest.main()
