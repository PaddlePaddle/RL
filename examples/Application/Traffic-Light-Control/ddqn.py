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
import paddle.nn as nn
import paddle.nn.functional as F

import copy
import numpy as np
import parl
from parl.utils.scheduler import LinearDecayScheduler


class DDQN(parl.Algorithm):
    def __init__(self, model, config):

        self.model = model

        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=40.0)
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=config['start_lr'],
            parameters=self.model.parameters(),
            grad_clip=clip)

        self.mse_loss = nn.MSELoss(reduction='mean')

        self.config = config
        self.lr_scheduler = LinearDecayScheduler(config['start_lr'],
                                                 config['max_train_steps'])
        self.lr = config['start_lr']
        self.target_model = copy.deepcopy(model)

        self.train_count = 0

        self.epsilon = self.config['epsilon']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay = self.config['epsilon_decay']

    def sample(self, obs):
        logits = self.model(obs)
        return logits

    def predict(self, obs):
        logits = self.model(obs)
        predict_actions = paddle.argmax(logits, axis=-1)
        return predict_actions

    def sync_target(self, decay=0.995):
        # soft update
        self.model.sync_weights_to(self.target_model, decay)

    def learn(self, obs, actions, dones, rewards, next_obs):
        # Update the Q network with the data sampled from the memory buffer.
        if self.train_count > 0 and self.train_count % self.config[
                'lr_decay_interval'] == 0:
            self.lr = self.lr_scheduler.step(
                step_num=self.config['lr_decay_interval'])
        terminal = dones
        actions_onehot = F.one_hot(
            actions.astype('int'), num_classes=self.model.act_dim)
        # shape of the pred_values: batch_size
        pred_values = paddle.sum(self.model(obs) * actions_onehot, axis=-1)
        greedy_action = self.model(next_obs).argmax(1)
        with paddle.no_grad():
            # target_model for evaluation, using the double DQN, the max_v_show just used for showing in the tensorborad
            max_v_show = paddle.max(self.target_model(next_obs), axis=-1)
            greedy_actions_onehot = F.one_hot(
                greedy_action, num_classes=self.model.act_dim)
            max_v = paddle.sum(
                self.target_model(next_obs) * greedy_actions_onehot, axis=-1)
            assert max_v.shape == rewards.shape
            target = rewards + (1 - terminal) * self.config['gamma'] * max_v
        Q_loss = 0.5 * self.mse_loss(pred_values, target)

        # optimize
        self.optimizer.clear_grad()
        Q_loss.backward()
        self.optimizer.step()
        self.train_count += 1
        if self.epsilon > self.epsilon_min and self.train_count % self.config[
                'epsilon_decay_interval'] == 0:
            self.epsilon *= self.epsilon_decay
        return Q_loss, pred_values.mean(), target.mean(), max_v_show.mean(
        ), self.train_count, self.lr, self.epsilon
