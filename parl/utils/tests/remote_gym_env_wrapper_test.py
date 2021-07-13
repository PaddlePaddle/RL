#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import threading
import time
import parl
import numpy as np
from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.client import disconnect
from parl.utils import logger, get_free_tcp_port
from parl.utils.env_utils import RemoteGymEnv


def float_equal(x1, x2):
    if np.abs(x1 - x2) < 1e-6:
        return True
    else:
        return False


# Test RemoteGymEnv
# for both discrete and continuous action space environment
class TestRemoteEnv(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_discrete_env_wrapper(self):
        logger.info("Running: test discrete_env_wrapper")
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        woker1 = Worker('localhost:{}'.format(port), 1)

        parl.connect('localhost:{}'.format(port))
        logger.info("Running: test discrete_env_wrapper: 1")

        env = RemoteGymEnv(env_name='MountainCar-v0')
        env.seed(1)
        env.render()

        obs, done = env.reset(), False
        observation_space = env.observation_space
        obs_space_high = observation_space.high
        obs_space_low = observation_space.low
        self.assertTrue(float_equal(obs_space_high[1], 0.07))
        self.assertTrue(float_equal(obs_space_low[0], -1.2))

        action_space = env.action_space
        act_dim = action_space.n
        self.assertEqual(act_dim, 3)

        # Run an episode with a random policy
        total_steps, episode_reward = 0, 0
        while not done:
            action = np.random.choice(act_dim)
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward
        logger.info('Episode done, total_steps {}, episode_reward {}'.format(
            total_steps, episode_reward))

        master.exit()
        woker1.exit()

    def test_continuous_env_wrapper(self):
        logger.info("Running: test continuous_env_wrapper")
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        woker1 = Worker('localhost:{}'.format(port), 1)

        parl.connect('localhost:{}'.format(port))
        logger.info("Running: test continuous_env_wrapper: 1")

        env = RemoteGymEnv(env_name='Pendulum-v0')
        env.seed(0)
        env.render()

        obs, done = env.reset(), False
        observation_space = env.observation_space
        obs_space_high = observation_space.high
        obs_space_low = observation_space.low
        self.assertTrue(float_equal(obs_space_high[1], 1.))
        self.assertTrue(float_equal(obs_space_low[1], -1.))

        action_space = env.action_space
        action_space_high = action_space.high
        action_space_low = action_space.low
        self.assertEqual(action_space_high, [2.])
        self.assertEqual(action_space_low, [-2.])

        # Run an episode with a random policy
        total_steps, episode_reward = 0, 0
        while not done:
            total_steps += 1
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
        logger.info('Episode done, total_steps {}, episode_reward {}'.format(
            total_steps, episode_reward))

        master.exit()
        woker1.exit()


if __name__ == '__main__':
    unittest.main()
