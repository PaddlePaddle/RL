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

import os
import paddle
import parl
from parl.utils import logger

import numpy as np
from tqdm import tqdm

from world import World
from environment import CityFlowEnv
from config import config

from obs_reward.presslight_obs_reward import PressureLightGenerator
from model.presslight_model import PressLightModel

from obs_reward.presslight_FRAP_obs_reward import PressureLightFRAPGenerator
from model.FRAP_model import PressLightFRAPModel

from obs_reward.sotl_obs import SotlGenerator
from agent.sotl_agent import SOTLAgent

from obs_reward.max_pressure_obs import MaxPressureGenerator
from agent.max_pressure_agent import MaxPressureAgent


def test_presslight(epsilon_num=1, episode_tag=300, is_replay=False):
    """
    test the env.
    """
    ########################################
    # creating the world and the env.
    ########################################
    logger.info('building the env...')
    world = World(
        config['config_path_name'],
        thread_num=config['thread_num'],
        yellow_phase_time=config['yellow_phase_time'])
    PLGenerator = PressureLightGenerator(world, config['obs_fns'],
                                         config['reward_fns'],
                                         config['is_only'], config['average'])
    obs_dims = PLGenerator.obs_dims
    env = CityFlowEnv(world, PLGenerator)
    obs = env.reset()
    act_dims = env.action_dims
    n_agents = env.n_agents
    if is_replay:
        env.world.eng.set_save_replay(is_replay)
        env.world.eng.set_replay_file('replay_presslight.txt')
    ########################################
    # creating the agents and
    # each agent has it own model.
    ########################################
    logger.info(
        'building {} agents, each agent has its model and algorithm...'.format(
            n_agents))
    models = [
        PressLightModel(obs_dims[i], act_dims[i], config['algo'])
        for i in range(n_agents)
    ]
    logger.info('successfully creating {} agents...'.format(n_agents))
    ########################################
    # loading the model from the ckpt model.
    ########################################
    for model_id, model in enumerate(models):
        model_path = os.path.join(
            config['save_dir'], 'agentid{}_episode_count{}.ckpt'.format(
                model_id, episode_tag))
        logger.info('agent: {}/{} loading model from {}...'.format(
            model_id + 1, n_agents, model_path))
        checkpoint = paddle.load(model_path)
        model.set_state_dict(checkpoint)
    ########################################
    # testing the model with env.
    ########################################
    total_avg_travel_time = []
    episode_count = 0

    with tqdm(total=epsilon_num, desc='[Testing Model]') as pbar:
        while episode_count < epsilon_num:
            step_count = 0
            while step_count < config['metric_period']:
                actions = []
                for agent_id, ob in enumerate(obs):
                    ob = ob.reshape(1, -1)
                    ob = paddle.to_tensor(ob, dtype='float32')
                    action = models[agent_id](ob)
                    action = action.numpy()
                    action = np.argmax(action)
                    actions.append(action)
                actions = np.array(actions)
                for _ in range(config['action_interval']):
                    step_count += 1
                    next_obs, rewards, dones, _ = env.step(actions)
                obs = next_obs
                if step_count % 200 == 0:
                    logger.info('esipode:{}, step_count:{}'.format(
                        episode_count, step_count))
            episode_count += 1
            avg_travel_time = env.world.eng.get_average_travel_time()

            logger.info('esipode:{}, avg_time:{}'.format(
                episode_count, avg_travel_time))
            total_avg_travel_time.append(avg_travel_time)
            obs = env.reset(seed=True)
            pbar.update(1)
    return total_avg_travel_time


def test_sotl(epsilon_num=1, is_replay=False):
    """
    test the env.
    """
    ########################################
    # creating the world and the env.
    ########################################
    logger.info('building the env...')
    world = World(
        config['config_path_name'],
        thread_num=config['thread_num'],
        yellow_phase_time=config['yellow_phase_time'])
    SLGenerator = SotlGenerator(world)
    env = CityFlowEnv(world, SLGenerator)
    obs = env.reset()
    act_dims = env.action_dims
    n_agents = env.n_agents
    if is_replay:
        env.world.eng.set_save_replay()
        env.world.eng.set_replay_file('replay_sotl.txt')
    ########################################
    # creating the agents.
    ########################################
    agent = SOTLAgent(world)

    ########################################
    # testing the agent with env.
    ########################################
    total_avg_travel_time = []
    episode_count = 0
    with tqdm(total=epsilon_num, desc='[Testing Model]') as pbar:
        while episode_count < epsilon_num:
            step_count = 0
            while step_count < config['metric_period']:
                actions = agent.predict(obs)
                step_count += 1
                next_obs, rewards, dones, _ = env.step(actions)
                obs = next_obs
                if step_count % 200 == 0:
                    logger.info('esipode:{}, step_count:{}'.format(
                        episode_count, step_count))
            episode_count += 1
            avg_travel_time = env.world.eng.get_average_travel_time()
            logger.info('esipode:{}, avg_time:{}'.format(
                episode_count, avg_travel_time))
            total_avg_travel_time.append(avg_travel_time)
            obs = env.reset(seed=True)
            pbar.update(1)


def test_max_pressure(epsilon_num=1, is_replay=False):
    """
    test the env.
    """
    ########################################
    # creating the world and the env.
    ########################################
    logger.info('building the env...')
    logger.info('loading config from {}'.format(config['config_path_name']))
    world = World(
        config['config_path_name'],
        thread_num=config['thread_num'],
        yellow_phase_time=config['yellow_phase_time'])
    MPGenerator = MaxPressureGenerator(world)
    env = CityFlowEnv(world, MPGenerator)
    obs = env.reset()
    act_dims = env.action_dims
    n_agents = env.n_agents
    if is_replay:
        env.world.eng.set_save_replay(True)
        env.world.eng.set_replay_file('replay_maxpressure.txt')
    ########################################
    # creating the agents.
    ########################################
    agent = MaxPressureAgent(world)
    ########################################
    # testing the agent with env.
    ########################################
    total_avg_travel_time = []
    episode_count = 0
    with tqdm(total=epsilon_num, desc='[Testing Model]') as pbar:
        while episode_count < epsilon_num:
            step_count = 0
            while step_count < config['metric_period']:
                actions = agent.predict(obs)
                yellow_time = config['yellow_phase_time']
                yellow_time = 0
                for _ in range(config['action_interval'] + yellow_time):
                    step_count += 1
                    next_obs, rewards, dones, _ = env.step(actions)
                obs = next_obs
                if step_count % 200 == 0:
                    logger.info('esipode:{}, step_count:{}'.format(
                        episode_count, step_count))
            episode_count += 1
            avg_travel_time = env.world.eng.get_average_travel_time()
            logger.info('esipode:{}, avg_time:{}'.format(
                episode_count, avg_travel_time))
            total_avg_travel_time.append(avg_travel_time)
            obs = env.reset(seed=True)
            pbar.update(1)
    return total_avg_travel_time


def test_FRAP_light(epsilon_num=1, episode_tag=300, is_replay=False):
    '''
    test the env.
    '''
    ########################################
    # creating the world and the env.
    ########################################
    logger.info('building the env...')
    world = World(
        config['config_path_name'],
        thread_num=config['thread_num'],
        yellow_phase_time=config['yellow_phase_time'])

    PLGenerator = PressureLightFRAPGenerator(world, config['obs_fns'],
                                             config['reward_fns'])
    relation_constants = PLGenerator.generate_relation()

    obs_dims = PLGenerator.obs_dims
    env = CityFlowEnv(world, PLGenerator)
    obs = env.reset()
    act_dims = env.action_dims
    n_agents = env.n_agents
    if is_replay:
        env.world.eng.set_save_replay(is_replay)
        env.world.eng.set_replay_file('replay_presslight.txt')
    ########################################
    # creating the agents and
    # each agent has it own model.
    ########################################
    logger.info(
        'building {} agents, each agent has its model and algorithm...'.format(
            n_agents))
    relation_constant = paddle.to_tensor(
        relation_constants[0], dtype='float32')
    relation_constant = relation_constant.astype('int')
    models = [
        PressLightFRAPModel(
            obs_dims[i], act_dims[i], constant=relation_constant)
        for i in range(n_agents)
    ]
    logger.info('successfully creating {} agents...'.format(n_agents))
    ########################################
    # loading the model
    ########################################
    for model_id, model in enumerate(models):
        model_path = os.path.join(
            config['save_dir'], 'agentid{}_episode_count{}.ckpt'.format(
                model_id, episode_tag))
        logger.info('agent: {}/{} loading model from {}...'.format(
            model_id + 1, n_agents, model_path))
        checkpoint = paddle.load(model_path)
        model.set_state_dict(checkpoint)
    ########################################
    # testing the model with env.
    ########################################
    total_avg_travel_time = []
    episode_count = 0

    with tqdm(total=epsilon_num, desc='[Testing Model]') as pbar:
        while episode_count < epsilon_num:
            step_count = 0
            while step_count < config['metric_period']:
                actions = []
                for agent_id, ob in enumerate(obs):
                    ob = ob.reshape(1, -1)
                    ob = paddle.to_tensor(ob, dtype='float32')
                    action = models[agent_id](ob)
                    action = action.detach().cpu().numpy()
                    action = np.argmax(action)
                    actions.append(action)
                actions = np.array(actions)
                for _ in range(config['action_interval']):
                    step_count += 1
                    next_obs, rewards, dones, _ = env.step(actions)
                obs = next_obs
                if step_count % 200 == 0:
                    logger.info('esipode:{}, step_count:{}'.format(
                        episode_count, step_count))
            episode_count += 1
            avg_travel_time = env.world.eng.get_average_travel_time()
            logger.info('esipode:{}, avg_time:{}'.format(
                episode_count, avg_travel_time))
            total_avg_travel_time.append(avg_travel_time)
            obs = env.reset(seed=True)
            pbar.update(1)
    return total_avg_travel_time


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path_name',
        default='./scenarios/config_hz_1.json',
        type=str,
        help='config path')
    parser.add_argument(
        '--is_test_frap', default=True, type=bool, help='test algorithm')
    parser.add_argument(
        '--result_name', default='4_4', type=str, help='result path')
    parser.add_argument(
        '--save_dir', default='./save_model', type=str, help='config path')
    parser.add_argument(
        '--episode_tag', default=300, type=int, help='episode_tag')

    args = parser.parse_args()

    config['config_path_name'] = args.config_path_name
    config['save_dir'] = args.save_dir
    if args.is_test_frap:
        results = test_FRAP_light(
            is_replay=False, episode_tag=args.episode_tag)
        path_name = 'result_frap'
    else:
        results = test_presslight(
            is_replay=False, episode_tag=args.episode_tag)
        path_name = 'result_max_pressure'
    # result_sotl = test_sotl()
    result_max_pressure = test_max_pressure(is_replay=False)

    result_path = path_name + '/{}'.format(args.result_name)
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, 'avgtime.txt'), 'w') as f:
        if args.is_test_frap:
            f.writelines('result_FRAP: ')
        else:
            f.writelines('result_presslight: ')
        f.writelines(str(results[0]))
        f.writelines('\n')
        f.writelines('result_max_pressure: ')
        f.writelines(str(result_max_pressure[0]))
