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

import torch
import torch.nn.functional as F
from copy import deepcopy
import parl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
Algorithm of DDPG: describes the mechanism of updating parameters in Model

Contributions: predict(obs): return act         # choose action based on actor net
               learn(replay_memory_batch)       # 
               sync_target()                    # update networks
'''
__all__ = ['DDPG']


class DDPG(parl.Algorithm):
    def __init__(
            self,
            model,
            max_action,
            discount=None,
            tau=None,
            actor_lr=None,
            critic_lr=None,
            decay=None,
            policy_freq=1,
    ):  # Frequency of delayed policy updates
        assert isinstance(discount, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.decay = decay
        self.policy_freq = policy_freq
        self.total_it = 0

        self.model = model.to(device)
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = torch.optim.Adam(self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.model.get_critic_params(), lr=critic_lr)

    def predict(self, obs):
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, ternimal):

        self.total_it += 1
        # Compute the target Q value
        target_Q = self.target_model.critic_model(next_obs, self.target_model.actor_model(next_obs))
        target_Q = reward + (ternimal * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.model.critic_model(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the frozen target models
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.model.critic_model(obs, self.model.actor_model(obs)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.sync_target(decay=self.decay)

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_((1 - decay) * param.data + decay * target_param.data)