# Third party code
#
# The following code is mainly referenced, modified and copied from:
# https://github.com/zhc134/tlc-baselines and https://github.com/gjzheng93/tlc-baseline

import numpy as np


class MaxPressureAgent(object):
    """
    Agent using MaxPressure method to control traffic light
    """

    def __init__(self, world):
        self.world = world

    def predict(self, lane_vehicle_count):
        actions = []
        for I_id, I in enumerate(self.world.intersections):
            action = I.current_phase
            max_pressure = None
            action = -1
            for phase_id in range(len(I.phases)):
                pressure = sum([
                    lane_vehicle_count[start] - lane_vehicle_count[end]
                    for start, end in I.phase_available_lanelinks[phase_id]
                ])
                if max_pressure is None or pressure > max_pressure:
                    action = phase_id
                    max_pressure = pressure
            actions.append(action)
        return np.array(actions)
