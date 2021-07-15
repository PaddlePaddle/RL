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


class PressLightFRAPModel(parl.Model):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 embedding_size=4,
                 constant=None,
                 algo='DQN'):

        super(PressLightFRAPModel, self).__init__()
        self.constant = constant
        self.phase_lanes_dim = (obs_dim - act_dim) // act_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Assert the input of phase is one-hot.
        self.current_phase_embedding = nn.Embedding(2, embedding_size)
        self.relation_embedding = nn.Embedding(2, embedding_size)

        relation_dim = 10
        self.relation_conv = nn.Conv2D(
            embedding_size, relation_dim, kernel_size=1, stride=1, padding=0)

        self.d_fc = nn.Linear(self.phase_lanes_dim, embedding_size)

        self.lane_dim = 16
        self.lane_fc = nn.Linear(embedding_size * 2, self.lane_dim)
        self.lane_conv = nn.Conv2D(
            self.lane_dim * 2,
            relation_dim,
            kernel_size=1,
            stride=1,
            padding=0)
        hidden_size = 10
        self.hidden_conv = nn.Conv2D(
            relation_dim, hidden_size, kernel_size=1, stride=1, padding=0)
        self.before_merge = nn.Conv2D(
            hidden_size, 1, kernel_size=1, stride=1, padding=0)

        self.algo = algo

    def forward(self, x):

        batch_size = x.shape[0]
        # The cur_phase is one-hot vector and only contains 0/1.
        cur_phase = x[:, self.obs_dim - self.act_dim:].astype('int')
        # cur_phase_em shape:[batch, act_dim, embedding_size]
        cur_phase_em = self.current_phase_embedding(cur_phase)

        # Constant and relation_embedding's shape:[batchsize, constant.shape[0], constant.shape[1], 4]
        constant = paddle.tile(self.constant, (batch_size, 1, 1))
        relation_embedding = self.relation_embedding(constant)
        # From NHWC to NCHW
        relation_embedding = paddle.transpose(
            relation_embedding, perm=[0, 3, 1, 2])
        relation_conv = self.relation_conv(relation_embedding)

        # The x_lane_phases contain lane vehicle nums of each phase,
        # there may be two or more lanes can pass because the phase set the lanes to green,
        # and the obs may sightly different to the origin paper, but it may be not affect the fianl result in our experiment.
        x_lane_phases = paddle.reshape(
            x[:, :self.obs_dim - self.act_dim],
            [-1, self.act_dim, self.phase_lanes_dim])
        # x_lane_phases_feature shape: [batch_size, act_dim, embedding_size]
        x_lane_phases_feature = nn.Sigmoid()(self.d_fc(x_lane_phases))
        list_phase_pressure = []
        for i in range(self.act_dim):
            # concat the embedding features
            p1_concat = paddle.concat(
                (x_lane_phases_feature[:, i], cur_phase_em[:, i]), axis=-1)
            add_feature = nn.Sigmoid()(self.lane_fc(p1_concat))
            list_phase_pressure.append(add_feature)

        list_phase_pressure_recomb = []
        for i in range(self.act_dim):
            for j in range(self.act_dim):
                if i != j:
                    list_phase_pressure_recomb.append(
                        paddle.concat(
                            (list_phase_pressure[i], list_phase_pressure[j]),
                            axis=-1))
        list_phase_pressure_recomb = paddle.stack(list_phase_pressure_recomb)
        list_phase_pressure_recomb = paddle.transpose(
            list_phase_pressure_recomb, perm=[1, 0, 2])
        # list_phase_pressure_recomb shape: [batch_size, self.act_dim*self.act_dim-1, 32]
        list_phase_pressure_recomb = paddle.reshape(
            list_phase_pressure_recomb,
            (-1, self.act_dim, self.act_dim - 1, self.lane_dim * 2))
        list_phase_pressure_recomb = paddle.transpose(
            list_phase_pressure_recomb, perm=[0, 3, 1, 2])
        lane_conv = self.lane_conv(list_phase_pressure_recomb)

        combine_feature = paddle.multiply(lane_conv, relation_conv)

        hidden_layer = self.hidden_conv(combine_feature)
        before_merge = self.before_merge(hidden_layer)
        before_merge = paddle.reshape(before_merge,
                                      (-1, self.act_dim, self.act_dim - 1))
        q_values = paddle.sum(before_merge, axis=-1)
        assert q_values.shape[-1] == self.act_dim
        return q_values
