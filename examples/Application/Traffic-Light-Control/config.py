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

config = {

    #==========  env config ==========
    'config_path_name':
    './scenarios/config_hz_1.json',  # note that the path of the data can be modified in the json file.
    'thread_num': 8,
    'obs_fns': ['lane_count'],
    'reward_fns': ['pressure'],
    'is_only': False,
    'average': None,
    'action_interval': 10,
    'metric_period': 3600,  #3600
    'yellow_phase_time': 5,

    #==========  learner config ==========
    'gamma': 0.85,  # also can be set to 0.95
    'epsilon': 0.9,
    'epsilon_min': 0.2,
    'epsilon_decay': 0.99,
    'start_lr': 0.00025,
    'episodes': 200 + 100,
    'algo': 'DQN',  # DQN
    'max_train_steps': int(1e6),
    'lr_decay_interval': 100,
    'epsilon_decay_interval': 100,
    'sample_batch_size':
    2048,  # also can be set to 32, which doesn't matter much.
    'learn_freq': 2,  # update parameters every 2 or 5 steps
    'decay': 0.995,  # soft update of double DQN
    'reward_normal_factor': 4,  # rescale the rewards, also can be set to 20,
    'train_count_log': 5,  # add to the tensorboard
    'is_show_log': False,  # print in the screen
    'step_count_log': 1000,

    # save checkpoint frequent episode
    'save_rate': 100,
    'save_dir': './save_model/presslight',
    'train_log_dir': './train_log/presslight',
    'save_dir': './save_model/presslight4*4',
    'train_log_dir': './train_log/presslight4*4',

    # memory config
    'memory_size': 20000,
    'begin_train_mmeory_size': 3000
}
