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
"""
The following code are referenced and modified from:
https://github.com/gjzheng93/frap-pub and https://github.com/zhc134/tlc-baselines
"""
import numpy as np


class PressureLightFRAPGenerator(object):
    """PressureLightFRAPGenerator

    Args:
        world (object): World used by this Generator.
        fns_obs: functions to get the obs.
        fns_reward: functions to get the rewards.
    """

    def __init__(self, world, fns_obs, fns_reward):

        self.world = world
        self.fns_obs = fns_obs
        # Get all the intersections, because that each intersection is one agent.
        self.Is = self.world.intersections
        # Get lanes of intersections, with the order of the list is same to the self.Is.
        self.all_intersections_lanes = []
        # May be the dim of each intersection can be different? Assert the all the agents have the same dims here.
        self.obs_dims = []
        for I in self.Is:
            # each intersection's lane_ids is saved in the lanes, and the infos needed such as the lane vehicle num of obs can be got from the lane_ids here.
            lanes = []
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
            available_lanelinks = I.phase_available_lanelinks[0]
            # phase_available_lanelinks of each phase contains start_end_lanelink_pair,
            # here we use the vehicle nums of the start_end_lanelink_pair as the feature,
            # so the dim of the obs is :len(I.phases)*len(available_lanelinks)*2  also plus the len(I.phases).
            # which may be slight different to the paper, but many other papers using the different feature and also get the better results,
            # and you can modify the feature_dim here and below.
            self.obs_dims.append(
                len(I.phases) * len(available_lanelinks) * 2 + len(I.phases))

        # subscribe functions for obs and reward
        self.world.subscribe(self.fns_obs)
        self.world.subscribe(fns_reward)
        self.fns_reward = fns_reward

    def generate_relation(self):
        """
        getting the confilt relation matrix, which can only use when the act_dim is 8 or 4.
        """
        relations_all = []
        for I in self.Is:
            relations = []
            num_phase = len(I.phases)
            if num_phase == 8:
                for p1 in I.phase_available_roadlinks:
                    zeros = [0, 0, 0, 0, 0, 0, 0]
                    count = 0
                    for p2 in I.phase_available_roadlinks:
                        if p1 == p2:
                            # That means that the two phase have one same direction.
                            continue
                        if len(list(set(p1 + p2))) == 3:
                            zeros[count] = 1
                        count += 1
                    relations.append(zeros)
                relations = np.array(relations).reshape((8, 7))
            elif num_phase == 4:
                relations = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0],
                                      [0, 0, 0]]).reshape((4, 3))
            else:
                assert 0
            relations_all.append(relations)
        return np.array(relations_all)

    def generate_phase_pairs(self):
        """
        pairs road set to green by the phase, each phase may set 2 roads to green light.
        """
        phase_available_roadlinks_all = []
        for I in self.Is:
            phase_available_roadlinks_all.append(I.phase_available_roadlinks)
        return np.array(phase_available_roadlinks_all)

    def generate_obs(self):
        """
        return: numpy array of all the intersections obs
        """
        # get all the infos for calc the obs of each intersections
        results = [self.world.get_info(fn) for fn in self.fns_obs]
        result = results[0]
        cur_phases = [I.current_phase for I in self.Is]
        ret_all = []
        # only get the vehilce nums, [I_num, phase_num * dim]
        all_ret = []
        for I_id, I in enumerate(self.Is):
            phase_lane_vehicle_num = []
            phase_onehot = [0 for _ in range(len(I.phases))]
            for phase_id in range(len(I.phases)):
                available_lanelinks = I.phase_available_lanelinks[phase_id]
                for start_end_lanelink_pair in available_lanelinks:
                    for lane_id in start_end_lanelink_pair:
                        # append the lane vehicle num for each lane_id in the available_lanelinks, both start and end road.
                        phase_lane_vehicle_num.append(result[lane_id])
            phase_onehot[cur_phases[I_id]] = 1
            phase_lane_vehicle_num.extend(phase_onehot)
            # Note that the len(phase_lane_vehicle_num) that should be equal to the self.obs_dims[I_id] above.
            all_ret.append(phase_lane_vehicle_num)
        all_ret = np.array(all_ret)
        return all_ret

    def generate_reward(self):
        """
        getting the reward of each intersections.
        """
        pressures = self.world.get_info(self.fns_reward[0])
        rewards = []
        for I in self.world.intersections:
            rewards.append(-pressures[I.id])
        return rewards
