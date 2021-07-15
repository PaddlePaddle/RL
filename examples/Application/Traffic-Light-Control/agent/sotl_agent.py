# Third party code
#
# The following code is mainly referenced, modified and copied from:
# https://github.com/zhc134/tlc-baselines and https://github.com/gjzheng93/tlc-baseline

import numpy as np


class SOTLAgent(object):
    """
    Agent using Self-organizing Traffic Light(SOTL) Control method to control traffic light.
    Note that different t_min, min_green_vehicle and max_red_vehicle may cause different results, which may not fair to compare to others.
    """

    def __init__(self, world, t_min=3, min_green_vehicle=20,
                 max_red_vehicle=5):
        self.world = world
        # the minimum duration of time of one phase
        self.t_min = t_min
        # some threshold to deal with phase requests
        self.min_green_vehicle = min_green_vehicle  # 10
        self.max_red_vehicle = max_red_vehicle  # 30
        self.action_dims = []
        for i in self.world.intersections:
            self.action_dims.append(len(i.phases))

    def predict(self, lane_waiting_count):
        actions = []
        for I_id, I in enumerate(self.world.intersections):
            action = I.current_phase
            if I.current_phase_time >= self.t_min:
                num_green_vehicles = sum([
                    lane_waiting_count[lane]
                    for lane in I.phase_available_startlanes[I.current_phase]
                ])
                num_red_vehicles = sum(
                    [lane_waiting_count[lane] for lane in I.startlanes])
                num_red_vehicles -= num_green_vehicles
                if num_green_vehicles <= self.min_green_vehicle and num_red_vehicles > self.max_red_vehicle:
                    action = (action + 1) % self.action_dims[I_id]
            actions.append(action)
        return np.array(actions)

    def get_reward(self):
        return None
