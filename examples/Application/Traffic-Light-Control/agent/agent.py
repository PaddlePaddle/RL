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

import paddle
import parl
import numpy as np


class Agent(parl.Agent):
    def __init__(self, algorithm, config):
        super(Agent, self).__init__(algorithm)

        self.config = config
        self.epsilon = self.config['epsilon']

    def sample(self, obs):
        # The epsilon-greedy action selector.
        obs = paddle.to_tensor(obs, dtype='float32')
        logits = self.alg.sample(obs)
        act_dim = logits.shape[-1]
        act_values = logits.numpy()
        actions = np.argmax(act_values, axis=-1)
        for i in range(obs.shape[0]):
            if np.random.rand() <= self.epsilon:
                actions[i] = np.random.randint(0, act_dim)
        return actions

    def predict(self, obs):

        obs = paddle.to_tensor(obs, dtype='float32')
        predict_actions = self.alg.predict(obs)
        return predict_actions.numpy()

    def learn(self, obs, actions, dones, rewards, next_obs):

        obs = paddle.to_tensor(obs, dtype='float32')
        actions = paddle.to_tensor(actions, dtype='float32')
        dones = paddle.to_tensor(dones, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        rewards = paddle.to_tensor(rewards, dtype='float32')

        Q_loss, pred_values, target_values, max_v_show_values, train_count, lr, epsilon = self.alg.learn(
            obs, actions, dones, rewards, next_obs)

        self.alg.sync_target(decay=self.config['decay'])
        self.epsilon = epsilon

        return Q_loss.numpy(), pred_values.numpy(), target_values.numpy(
        ), max_v_show_values.numpy(), train_count, lr, epsilon
