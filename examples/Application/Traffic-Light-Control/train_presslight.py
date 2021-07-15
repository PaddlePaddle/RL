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
from parl.utils import logger, ReplayMemory

import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from world import World
from obs_reward.presslight_obs_reward import PressureLightGenerator
from config import config
from environment import CityFlowEnv
from model.presslight_model import PressLightModel
from ddqn import DDQN
from agent.agent import Agent


def log_metrics(summary, datas, buffer_total_size, is_show=False):
    """ Log metrics 
        """
    Q_loss, pred_values, target_values, max_v_show_values, train_count, lr, epsilon = datas
    metric = {
        'q_loss': Q_loss,
        'pred_values': pred_values,
        'target_values': target_values,
        'max_v_show_values': max_v_show_values,
        'lr': lr,
        'epsilon': epsilon,
        'memory_size': buffer_total_size,
        'train_count': train_count
    }
    if is_show:
        logger.info(metric)
    for key in metric:
        if key != 'train_count':
            summary.add_scalar(key, metric[key], train_count)


def main():
    """
    each intersection has each own model.
    """
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
    episode_count = 0
    step_forward = 0
    ####################
    act_dims = env.action_dims
    n_agents = env.n_agents
    logger.info(
        'creating {} replay_buffers for {} agents, each agent has one replay buffer.'
        .format(n_agents, n_agents))
    replay_buffers = [
        ReplayMemory(config['memory_size'], obs_dims[i], 0)
        for i in range(n_agents)
    ]
    ####################
    logger.info(
        'building {} agents, each agent has its model and algorithm...'.format(
            n_agents))
    models = [
        PressLightModel(obs_dims[i], act_dims[i], config['algo'])
        for i in range(n_agents)
    ]
    algorithms = [DDQN(model, config) for model in models]
    agents = [Agent(algorithm, config) for algorithm in algorithms]
    logger.info('successfully creating {} agents...'.format(n_agents))
    ####################
    # tensorboard list
    summarys = [
        SummaryWriter(os.path.join(config['train_log_dir'], str(agent_id)))
        for agent_id in range(n_agents)
    ]

    ###################
    episodes_rewards = np.zeros(n_agents)
    ###################
    with tqdm(total=config['episodes'], desc='[Training Model]') as pbar:
        while episode_count <= config['episodes']:
            step_count = 0
            while step_count < config['metric_period']:
                actions = []
                for agent_id, ob in enumerate(obs):
                    ob = ob.reshape(1, -1)
                    action = agents[agent_id].sample(ob)
                    actions.append(action[0])
                actions = np.array(actions)
                rewards_list = []
                for _ in range(config['action_interval']):
                    step_count += 1
                    next_obs, rewards, dones, _ = env.step(actions)
                    rewards_list.append(rewards)
                rewards = np.mean(
                    rewards_list, axis=0) / config['reward_normal_factor']
                # calc the episodes_rewards and will add it to the tensorboard
                assert len(episodes_rewards) == len(rewards)
                episodes_rewards += rewards
                for agent_id, replay_buffer in enumerate(replay_buffers):
                    replay_buffers[agent_id].append(
                        obs[agent_id], actions[agent_id], rewards[agent_id],
                        next_obs[agent_id], dones[agent_id])
                step_forward += 1
                obs = next_obs
                if len(replay_buffers[0]) >= config[
                        'begin_train_mmeory_size'] and step_forward % config[
                            'learn_freq'] == 0:
                    for agent_id, agent in enumerate(agents):
                        sample_data = replay_buffers[agent_id].sample_batch(
                            config['sample_batch_size'])
                        train_obs, train_actions, train_rewards, train_next_obs, train_terminals = sample_data

                        Q_loss, pred_values, target_values, max_v_show_values, train_count, lr, epsilon = \
                            agent.learn(train_obs, train_actions, train_terminals, train_rewards, train_next_obs)
                        datas = [
                            Q_loss, pred_values, target_values,
                            max_v_show_values, train_count, lr, epsilon
                        ]
                        # tensorboard
                        if train_count % config['train_count_log'] == 0:
                            log_metrics(summarys[agent_id], datas,
                                        step_forward)
                if step_count % config['step_count_log'] == 0 and config[
                        'is_show_log']:
                    logger.info('episode_count: {}, step_count: {}, buffer_size: {}, buffer_size_total_size: {}.'\
                        .format(episode_count, step_count, len(replay_buffers[0]), step_forward))

            episode_count += 1
            avg_travel_time = env.world.eng.get_average_travel_time()
            obs = env.reset()
            for agent_id, summary in enumerate(summarys):
                summary.add_scalar('episodes_reward',
                                   episodes_rewards[agent_id], episode_count)
                # the avg travel time is same for all agents.
                summary.add_scalar('average_travel_time', avg_travel_time,
                                   episode_count)
            logger.info('episode_count: {}, average_travel_time: {}.'.format(
                episode_count, avg_travel_time))
            # reset to zeros
            episodes_rewards = np.zeros(n_agents)
            # save the model
            if episode_count % config['save_rate'] == 0:
                for agent_id, agent in enumerate(agents):
                    save_path = "{}/agentid{}_episode_count{}.ckpt".format(
                        config['save_dir'], agent_id, episode_count)
                    agent.save(save_path)
            pbar.update(1)


def main_all():
    """
    all intersections share one model.
    """
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
    episode_count = 0
    step_forward = 0
    ####################
    act_dims = env.action_dims
    n_agents = env.n_agents
    replay_buffer = ReplayMemory(config['memory_size'] * n_agents, obs_dims[0],
                                 0)
    ###################
    model = PressLightModel(obs_dims[0], act_dims[0], config['algo'])
    algorithm = DDQN(model, config)
    agent = Agent(algorithm, config)
    logger.info('successfully creating the agent...')
    ###################
    # tensorboard list
    ###################
    summary = SummaryWriter(os.path.join(config['train_log_dir'], 'same'))
    ###################
    # train the model
    ###################
    episodes_rewards = np.zeros(n_agents)
    with tqdm(total=config['episodes'], desc='[Training Model]') as pbar:
        while episode_count <= config['episodes']:
            step_count = 0
            while step_count < config['metric_period']:
                actions = agent.sample(obs)
                rewards_list = []
                for _ in range(config['action_interval']):
                    step_count += 1
                    next_obs, rewards, dones, _ = env.step(actions)
                    rewards_list.append(rewards)
                rewards = np.mean(
                    rewards_list, axis=0) / config['reward_normal_factor']
                # calc the episodes_rewards and will add it to the tensorboard
                assert len(episodes_rewards) == len(rewards)
                episodes_rewards += rewards
                for agent_id in range(n_agents):
                    replay_buffer.append(obs[agent_id], actions[agent_id],
                                         rewards[agent_id], next_obs[agent_id],
                                         dones[agent_id])
                step_forward += 1
                obs = next_obs
                if len(replay_buffer) >= config[
                        'begin_train_mmeory_size'] and step_forward % config[
                            'learn_freq'] == 0:
                    sample_data = replay_buffer.sample_batch(
                        config['sample_batch_size'])
                    train_obs, train_actions, train_rewards, train_next_obs, train_terminals = sample_data
                    Q_loss, pred_values, target_values, max_v_show_values, train_count, lr, epsilon = \
                        agent.learn(train_obs, train_actions, train_terminals, train_rewards, train_next_obs)
                    datas = [
                        Q_loss, pred_values, target_values, max_v_show_values,
                        train_count, lr, epsilon
                    ]
                    # tensorboard
                    if train_count % config['train_count_log'] == 0:
                        log_metrics(summary, datas, step_forward)
            episode_count += 1
            avg_travel_time = env.world.eng.get_average_travel_time()
            obs = env.reset()
            # just calc the first agent's rewards for show.
            summary.add_scalar('episodes_reward', episodes_rewards[0],
                               episode_count)
            # the avg travel time is same for all agents.
            summary.add_scalar('average_travel_time', avg_travel_time,
                               episode_count)
            logger.info('episode_count: {}, average_travel_time: {}.'.format(
                episode_count, avg_travel_time))
            # reset to zeros
            episodes_rewards = np.zeros(n_agents)
            # save the model
            if episode_count % config['save_rate'] == 0:
                save_path = "{}/agentid{}_episode_count{}.ckpt".format(
                    config['save_dir'], '_same', episode_count)
                agent.save(save_path)
            pbar.update(1)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path_name',
        default='./scenarios/config_hz_1.json',
        type=str,
        help='config path')

    parser.add_argument(
        '--save_dir', default='./save_model', type=str, help='config path')

    parser.add_argument(
        '--is_share_model', default=False, type=bool, help='share_model')
    args = parser.parse_args()

    config['config_path_name'] = args.config_path_name
    config['save_dir'] = args.save_dir
    if args.is_share_model:
        main_all()
    else:
        main()
