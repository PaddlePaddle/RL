# Third party code
#
# The following code is mainly referenced, modified and copied from:
# https://github.com/zhc134/tlc-baselines and https://github.com/gjzheng93/tlc-baseline

import numpy as np


class MaxPressureGenerator(object):
    """
    Generate State based on statistics of lane vehicles.
    Parameters
    ----------
    world : World object
    fns_obs/fns_reward : list of statistics to get, currently support "lane_count", "lane_waiting_count" , "lane_waiting_time_count", "lane_delay" and "pressure"
    """

    def __init__(self, world, fns_obs='lane_count', fns_reward=None):

        self.world = world
        self.fns_obs = fns_obs
        # subscribe functions for obs and reward
        self.world.subscribe(self.fns_obs)
        self.fns_reward = fns_reward
        if self.fns_reward:
            self.world.subscribe(fns_reward)

    def generate_obs(self):
        """
        return: numpy array of all the intersections obs
        """
        lane_waiting_count = self.world.get_info(self.fns_obs)

        return lane_waiting_count

    def generate_reward(self):
        """
        getting the reward of each intersections
        """
        return None
