# Third party code
#
# The following code are copied or modified from:
# https://github.com/gjzheng93/tlc-baseline and https://github.com/zhc134/tlc-baselines

import gym
import numpy as np
import cityflow


class CityFlowEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.

    Parameters
    ----------
    world: World object
    obs_reward_generator(object): generator of the obs and rewards
    """

    def __init__(self, world, obs_reward_generator):

        self.world = world
        self.n_agents = len(self.world.intersection_ids)
        self.n = self.n_agents
        # agents action space dim, each roadnet file may have different action dims
        self.action_dims = []
        for i in self.world.intersections:
            self.action_dims.append(len(i.phases))
        self.action_space = gym.spaces.MultiDiscrete(self.action_dims)
        self.obs_reward_generator = obs_reward_generator

    def step(self, actions):
        """
        actions: list
        """
        assert len(actions) == self.n_agents
        self.world.step(actions)

        obs = self.obs_reward_generator.generate_obs()
        rewards = self.obs_reward_generator.generate_reward()
        dones = [False] * self.n_agents
        infos = {}
        return obs, rewards, dones, infos

    def reset(self, seed=False):
        self.world.reset(seed)
        obs = self.obs_reward_generator.generate_obs()
        return obs
