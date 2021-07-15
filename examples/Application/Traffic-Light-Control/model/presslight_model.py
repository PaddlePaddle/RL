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
import parl


class PressLightModel(parl.Model):
    def __init__(self, obs_dim, act_dim, algo='DQN'):
        super(PressLightModel, self).__init__()

        hid1_size = 20
        hid2_size = 20
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        embedding_size = 10
        self.current_phase_embedding = nn.Embedding(act_dim, embedding_size)

        self.algo = algo
        if self.algo == 'Dueling':
            self.fc1_adv = nn.Linear(obs_dim - 1, hid1_size)
            self.fc1_val = nn.Linear(obs_dim - 1, hid1_size)

            self.fc2_adv = nn.Linear(hid1_size, hid2_size)
            self.fc2_val = nn.Linear(hid1_size, hid2_size)

            self.fc3_adv = nn.Linear(hid2_size + embedding_size, self.act_dim)
            self.fc3_val = nn.Linear(hid2_size + embedding_size, 1)
        else:
            self.fc1 = nn.Linear(obs_dim - 1, hid1_size)
            self.fc2 = nn.Linear(hid1_size, hid2_size)
            self.fc3 = nn.Linear(hid2_size + embedding_size, self.act_dim)

    def forward(self, x):
        cur_phase = x[:, -1]
        cur_phase = cur_phase.astype('int')
        cur_phase_em = self.current_phase_embedding(cur_phase)
        x = x[:, :-1]
        if self.algo == 'Dueling':
            fc1_a = nn.ReLU()(self.fc1_adv(x))
            fc1_v = nn.ReLU()(self.fc1_val(x))

            fc2_a = nn.ReLU()(self.fc2_adv(fc1_a))
            fc2_v = nn.ReLU()(self.fc2_val(fc1_v))

            fc2_a = paddle.concat((fc2_a, cur_phase_em), axis=-1)
            fc2_v = paddle.concat((fc2_v, cur_phase_em), axis=-1)
            As = self.fc3_adv(fc2_a)
            V = self.fc3_val(fc2_v)
            Q = As + (V - As.mean(axis=1, keepdim=True))
        else:
            x1 = nn.ReLU()(self.fc1(x))
            x2 = nn.ReLU()(self.fc2(x1))
            x2 = paddle.concat((x2, cur_phase_em), axis=-1)
            Q = self.fc3(x2)
        return Q
