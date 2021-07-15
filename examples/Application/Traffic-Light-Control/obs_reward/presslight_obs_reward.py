# Third party code
#
# The following code are referenced, copied or modified from:
# https://github.com/zhc134/tlc-baselines and https://github.com/gjzheng93/tlc-baseline

import numpy as np


class PressureLightGenerator(object):
    """PressureLightGenerator

    Args:
        world (object): World used by this Generator.
        fns_obs: Functions to get the obs.
        fns_reward: Functions to get the rewards.
        in_only: Only the incoming roads or the all road.
        average: Whether the average nums or the num each load.
    """

    def __init__(self, world, fns_obs, fns_reward, in_only=False,
                 average=None):

        self.world = world
        self.fns_obs = fns_obs
        # get all the intersections
        self.Is = self.world.intersections
        # get lanes of intersections, with the order of the list of self.Is
        self.all_intersections_lanes = []
        self.obs_dims = []
        for I in self.Is:
            # each intersection's lane_ids is saved in the lanes, and the infos needed for obs can be got from the lane_ids here.
            lanes = []
            # road_ids
            if in_only:
                roads = I.in_roads
            else:
                roads = I.roads
            # get the lane_ids from the road_ids
            for road in roads:
                from_zero = (road["startIntersection"] == I.id
                             ) if self.world.RIGHT else (
                                 road["endIntersection"] == I.id)
                lanes.append([
                    road["id"] + "_" + str(i)
                    for i in range(len(road["lanes"]))[::(
                        1 if from_zero else -1)]
                ])
            # all the lanes of the all the intersections are saved in the self.all_intersections_lanes
            self.all_intersections_lanes.append(lanes)
            # calculate result dim of obs of each agents
            size = sum(len(x) for x in lanes)
            if average == "road":
                size = len(roads)
            elif average == "all":
                size = 1
            # In the pressure light len(self.fns_obs) is 1, and the curphase.
            self.obs_dims.append(len(self.fns_obs) * size + 1)
        # subscribe functions for obs and reward
        self.world.subscribe(self.fns_obs)

        self.world.subscribe(fns_reward)
        self.fns_reward = fns_reward
        self.average = average

    def generate_obs(self):
        """
        return: numpy array of all the intersections obs
        assert that each lane's dim is same.
        """
        # get all the infos for calc the obs of each intersections
        results = [self.world.get_info(fn) for fn in self.fns_obs]

        cur_phases = [I.current_phase for I in self.Is]
        ret_all = []
        for I_id, lanes in enumerate(self.all_intersections_lanes):
            ret = np.array([])
            for i in range(len(self.fns_obs)):
                result = results[i]
                fn_result = np.array([])
                for road_lanes in lanes:
                    road_result = []
                    for lane_id in road_lanes:
                        road_result.append(result[lane_id])
                    if self.average == "road" or self.average == "all":
                        road_result = np.mean(road_result)
                    else:
                        road_result = np.array(road_result)
                    fn_result = np.append(fn_result, road_result)
                if self.average == "all":
                    fn_result = np.mean(fn_result)
                ret = np.append(ret, fn_result)
                # append cur_phase in the last.
                ret = np.append(ret, cur_phases[I_id])
            ret_all.append(ret)
        return np.array(ret_all)

    def generate_reward(self):
        """
        getting the reward of each intersections, using the pressure.
        """
        pressures = self.world.get_info(self.fns_reward[0])
        rewards = []
        for I in self.world.intersections:
            rewards.append(-pressures[I.id])
        return rewards
