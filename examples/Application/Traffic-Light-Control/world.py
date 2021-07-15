# Third party code
#
# The following code are copied or modified from
# https://github.com/zhc134/tlc-baselines and https://github.com/gjzheng93/tlc-baseline

import json
import os.path as osp
import cityflow
import numpy as np
from math import atan2, pi
import sys


def _get_direction(road, out=True):
    if out:
        x = road["points"][1]["x"] - road["points"][0]["x"]
        y = road["points"][1]["y"] - road["points"][0]["y"]
    else:
        x = road["points"][-2]["x"] - road["points"][-1]["x"]
        y = road["points"][-2]["y"] - road["points"][-1]["y"]
    tmp = atan2(x, y)
    return tmp if tmp >= 0 else (tmp + 2 * pi)


class Intersection(object):
    def __init__(self, intersection, world, yellow_phase_time=5):

        self.id = intersection["id"]
        self.point = [intersection["point"]["x"], intersection["point"]["y"]]
        # using the world eng
        self.eng = world.eng
        # incoming and outgoing roads of each intersection, clock-wise order from North
        self.roads = []
        self.outs = []
        self.directions = []
        self.out_roads = None
        self.in_roads = None

        # links and phase information of each intersection
        self.roadlinks = []
        self.lanelinks_of_roadlink = []
        self.startlanes = []
        self.lanelinks = []
        self.phase_available_roadlinks = []
        self.phase_available_lanelinks = []
        self.phase_available_startlanes = []

        # define yellow phases, currently the default yellow phase is 0, so make sure the first phase of the roadnet is yellow phase
        self.yellow_phase_id = [0]
        # the default time of the yellow signal time is 5 seconds, you can change it to the real case.
        self.yellow_phase_time = yellow_phase_time

        # parsing links and phases
        for roadlink in intersection["roadLinks"]:
            self.roadlinks.append((roadlink["startRoad"], roadlink["endRoad"]))
            lanelinks = []
            for lanelink in roadlink["laneLinks"]:
                startlane = roadlink["startRoad"] + "_" + str(
                    lanelink["startLaneIndex"])
                self.startlanes.append(startlane)
                endlane = roadlink["endRoad"] + "_" + str(
                    lanelink["endLaneIndex"])
                lanelinks.append((startlane, endlane))
            self.lanelinks.extend(lanelinks)
            self.lanelinks_of_roadlink.append(lanelinks)

        self.startlanes = list(set(self.startlanes))

        phases = intersection["trafficLight"]["lightphases"]
        self.phases = [
            i for i in range(len(phases)) if not i in self.yellow_phase_id
        ]
        for i in self.phases:
            phase = phases[i]
            self.phase_available_roadlinks.append(phase["availableRoadLinks"])
            phase_available_lanelinks = []
            phase_available_startlanes = []
            for roadlink_id in phase["availableRoadLinks"]:
                lanelinks_of_roadlink = self.lanelinks_of_roadlink[roadlink_id]
                phase_available_lanelinks.extend(lanelinks_of_roadlink)
                for lanelinks in lanelinks_of_roadlink:
                    phase_available_startlanes.append(lanelinks[0])
            self.phase_available_lanelinks.append(phase_available_lanelinks)
            phase_available_startlanes = list(set(phase_available_startlanes))
            self.phase_available_startlanes.append(phase_available_startlanes)

        self.reset()

    def insert_road(self, road, out):

        self.roads.append(road)
        self.outs.append(out)
        self.directions.append(_get_direction(road, out))

    def sort_roads(self, RIGHT):

        order = sorted(
            range(len(self.roads)),
            key=
            lambda i: (self.directions[i], self.outs[i] if RIGHT else not self.outs[i])
        )
        self.roads = [self.roads[i] for i in order]
        self.directions = [self.directions[i] for i in order]
        self.outs = [self.outs[i] for i in order]
        self.out_roads = [self.roads[i] for i, x in enumerate(self.outs) if x]
        self.in_roads = [
            self.roads[i] for i, x in enumerate(self.outs) if not x
        ]

    def _change_phase(self, phase, interval):
        self.eng.set_tl_phase(self.id, phase)
        self._current_phase = phase
        self.current_phase_time = interval

    def step(self, action, interval):
        # if current phase is yellow, then continue to finish the yellow phase
        # recall self._current_phase means true phase id (including yellows)
        # self.current_phase means phase id in self.phases (excluding yellow)
        if self._current_phase in self.yellow_phase_id:
            if self.current_phase_time >= self.yellow_phase_time:
                self._change_phase(self.phases[self.action_before_yellow],
                                   interval)
                self.current_phase = self.action_before_yellow
            else:
                self.current_phase_time += interval
        else:
            if action == self.current_phase:
                self.current_phase_time += interval
            else:
                if self.yellow_phase_time > 0:
                    self._change_phase(self.yellow_phase_id[0], interval)
                    self.action_before_yellow = action
                else:
                    self._change_phase(action, interval)
                    self.current_phase = action

    def reset(self):
        # record phase info
        self.current_phase = 0  # phase id in self.phases (excluding yellow)
        self._current_phase = self.phases[
            0]  # true phase id (including yellow)
        self.eng.set_tl_phase(self.id, self._current_phase)
        self.current_phase_time = 0
        self.action_before_yellow = None


class World(object):
    """
    Create a CityFlow engine and maintain informations about CityFlow world
    """

    def __init__(self, cityflow_config, thread_num, yellow_phase_time=3):
        # loading the config and building the world..
        self.eng = cityflow.Engine(cityflow_config, thread_num=thread_num)
        with open(cityflow_config) as f:
            cityflow_config = json.load(f)
        self.roadnet = self._get_roadnet(cityflow_config)

        # vehicles moves on the right side, currently always set to true due to CityFlow's mechanism.
        self.RIGHT = True
        self.interval = cityflow_config["interval"]
        # get all non virtual intersections
        self.intersections = [
            i for i in self.roadnet["intersections"] if not i["virtual"]
        ]
        self.intersection_ids = [i["id"] for i in self.intersections]
        # create non-virtual Intersections
        print("creating intersections...")
        non_virtual_intersections = [
            i for i in self.roadnet["intersections"] if not i["virtual"]
        ]
        self.intersections = [
            Intersection(i, self, yellow_phase_time)
            for i in non_virtual_intersections
        ]
        self.intersection_ids = [i["id"] for i in non_virtual_intersections]
        self.id2intersection = {i.id: i for i in self.intersections}
        print("intersections created.")
        # id of all roads and lanes
        print("parsing roads...")
        self.all_roads = []
        self.all_lanes = []

        for road in self.roadnet["roads"]:
            self.all_roads.append(road["id"])
            i = 0
            for _ in road["lanes"]:
                self.all_lanes.append(road["id"] + "_" + str(i))
                i += 1

            iid = road["startIntersection"]
            if iid in self.intersection_ids:
                self.id2intersection[iid].insert_road(road, True)
            iid = road["endIntersection"]
            if iid in self.intersection_ids:
                self.id2intersection[iid].insert_road(road, False)

        for i in self.intersections:
            i.sort_roads(self.RIGHT)
        print("roads parsed.")

        # initializing info functions
        self.info_functions = {
            "vehicles": (lambda: self.eng.get_vehicles(include_waiting=True)),
            "lane_count": self.eng.get_lane_vehicle_count,
            "lane_waiting_count": self.eng.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.eng.get_lane_vehicles,
            "time": self.eng.get_current_time,
            "vehicle_distance": self.eng.get_vehicle_distance,
            "pressure": self.get_pressure,
            "lane_waiting_time_count": self.get_lane_waiting_time_count,
            "lane_delay": self.get_lane_delay,
            "vehicle_trajectory": self.get_vehicle_trajectory,
            "history_vehicles": self.get_history_vehicles
        }
        self.fns = []
        self.info = {}

        self.vehicle_waiting_time = {
        }  # key: vehicle_id, value: the waiting time of this vehicle since last halt.
        self.vehicle_trajectory = {
        }  # key: vehicle_id, value: [[lane_id_1, enter_time, time_spent_on_lane_1], ... , [lane_id_n, enter_time, time_spent_on_lane_n]]
        self.history_vehicles = set()

        print("world built successfully.")

    def get_pressure(self):
        vehicles = self.eng.get_lane_vehicle_count()
        pressures = {}
        for i in self.intersections:
            pressure = 0
            in_lanes = []
            for road in i.in_roads:
                from_zero = (
                    road["startIntersection"] == i.id) if self.RIGHT else (
                        road["endIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    in_lanes.append(road["id"] + "_" + str(n))
            out_lanes = []
            for road in i.out_roads:
                from_zero = (
                    road["endIntersection"] == i.id) if self.RIGHT else (
                        road["startIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    out_lanes.append(road["id"] + "_" + str(n))
            for lane in vehicles.keys():
                if lane in in_lanes:
                    pressure += vehicles[lane]
                if lane in out_lanes:
                    pressure -= vehicles[lane]
            pressures[i.id] = pressure
        return pressures

    def get_vehicle_lane(self):
        # get the current lane of each vehicle. {vehicle_id: lane_id}
        vehicle_lane = {}
        lane_vehicles = self.eng.get_lane_vehicles()
        for lane in self.all_lanes:
            for vehicle in lane_vehicles[lane]:
                vehicle_lane[vehicle] = lane
        return vehicle_lane

    def get_vehicle_waiting_time(self):
        # the waiting time of vehicle since last halt.
        vehicles = self.eng.get_vehicles(include_waiting=False)
        vehicle_speed = self.eng.get_vehicle_speed()
        for vehicle in vehicles:
            if vehicle not in self.vehicle_waiting_time.keys():
                self.vehicle_waiting_time[vehicle] = 0
            if vehicle_speed[vehicle] < 0.1:
                self.vehicle_waiting_time[vehicle] += 1
            else:
                self.vehicle_waiting_time[vehicle] = 0
        return self.vehicle_waiting_time

    def get_lane_waiting_time_count(self):
        # the sum of waiting times of vehicles on the lane since their last halt.
        lane_waiting_time = {}
        lane_vehicles = self.eng.get_lane_vehicles()
        vehicle_waiting_time = self.get_vehicle_waiting_time()
        for lane in self.all_lanes:
            lane_waiting_time[lane] = 0
            for vehicle in lane_vehicles[lane]:
                lane_waiting_time[lane] += vehicle_waiting_time[vehicle]
        return lane_waiting_time

    def get_lane_delay(self, speed_limit=11.11):
        # the delay of each lane: 1 - lane_avg_speed/speed_limit
        # set speed limit to 11.11 by default
        lane_vehicles = self.eng.get_lane_vehicles()
        lane_delay = {}
        lanes = self.all_lanes
        vehicle_speed = self.eng.get_vehicle_speed()

        for lane in lanes:
            vehicles = lane_vehicles[lane]
            lane_vehicle_count = len(vehicles)
            lane_avg_speed = 0.0
            for vehicle in vehicles:
                speed = vehicle_speed[vehicle]
                lane_avg_speed += speed
            if lane_vehicle_count == 0:
                lane_avg_speed = speed_limit
            else:
                lane_avg_speed /= lane_vehicle_count
            lane_delay[lane] = 1 - lane_avg_speed / speed_limit
        return lane_delay

    def get_vehicle_trajectory(self):

        # lane_id and time spent on the corresponding lane that each vehicle went through
        vehicle_lane = self.get_vehicle_lane()
        vehicles = self.eng.get_vehicles(include_waiting=False)
        for vehicle in vehicles:
            if vehicle not in self.vehicle_trajectory:
                self.vehicle_trajectory[vehicle] = [[
                    vehicle_lane[vehicle],
                    int(self.eng.get_current_time()), 0
                ]]
            else:
                if vehicle not in vehicle_lane.keys():
                    continue
                if vehicle_lane[vehicle] == self.vehicle_trajectory[vehicle][
                        -1][0]:
                    self.vehicle_trajectory[vehicle][-1][2] += 1
                else:
                    self.vehicle_trajectory[vehicle].append([
                        vehicle_lane[vehicle],
                        int(self.eng.get_current_time()), 0
                    ])
        return self.vehicle_trajectory

    def get_history_vehicles(self):

        self.history_vehicles.update(self.eng.get_vehicles())
        return self.history_vehicles

    def _get_roadnet(self, cityflow_config):
        roadnet_file = osp.join(cityflow_config["dir"],
                                cityflow_config["roadnetFile"])
        with open(roadnet_file) as f:
            roadnet = json.load(f)
        return roadnet

    def subscribe(self, fns):
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if not fn in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception("info function %s not exists" % fn)

    def step(self, actions=None):
        if actions is not None:
            for i, action in enumerate(actions):
                self.intersections[i].step(action, self.interval)
        self.eng.next_step()
        self._update_infos()

    def reset(self, seed):
        self.eng.reset(seed)
        for I in self.intersections:
            I.reset()
        self._update_infos()

    def _update_infos(self):
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()

    def get_info(self, info):
        return self.info[info]


if __name__ == "__main__":
    # testing the env.
    world = World("examples/config.json", thread_num=1)
    print(world.intersections[0].phase_available_startlanes)
