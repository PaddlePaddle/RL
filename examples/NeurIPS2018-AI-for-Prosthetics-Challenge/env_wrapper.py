import abc
import copy
import gym
import math
import numpy as np
from collections import OrderedDict
from osim.env import ProstheticsEnv
from parl.utils import logger

MAXTIME_LIMIT = 1000
ProstheticsEnv.time_limit = MAXTIME_LIMIT
FRAME_SKIP = None
FALL_PENALTY = 0


class RemoteEnv(gym.Wrapper):
    def __init__(self, env):
        env.metadata = {}
        env.action_space = None
        env.observation_space = None
        env.reward_range = None
        gym.Wrapper.__init__(self, env)
        self.remote_env = env
        self.first_time = True

    def step(self, act):
        return self.remote_env.env_step(act.tolist())

    def reset(self):
        if self.first_time:
            self.first_time = False
            return self.remote_env.env_create()
        obs = self.remote_env.env_reset()
        if not obs:
            return None
        return obs


def calc_vel_diff(state_desc):
    cur_vel_x = state_desc['body_vel']['pelvis'][0]
    cur_vel_z = state_desc['body_vel']['pelvis'][2]
    target_vel_x = state_desc['target_vel'][0]
    target_vel_z = state_desc['target_vel'][2]
    diff_vel_x = cur_vel_x - target_vel_x
    diff_vel_z = cur_vel_z - target_vel_z

    cur_vel = (cur_vel_x**2 + cur_vel_z**2)**0.5
    target_vel = (target_vel_x**2 + target_vel_z**2)**0.5
    diff_vel = cur_vel - target_vel

    target_theta = math.atan(-1.0 * target_vel_z / target_vel_x)
    # alone y axis
    cur_theta = state_desc['body_pos_rot']['pelvis'][1]
    diff_theta = cur_theta - target_theta

    return cur_vel_x, cur_vel_z, diff_vel_x, diff_vel_z, diff_vel, diff_theta


class ActionScale(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action, **kwargs):
        action = (np.copy(action) + 1.0) * 0.5
        action = np.clip(action, 0.0, 1.0)
        return self.env.step(action, **kwargs)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameSkip(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.frame_skip = k
        global FRAME_SKIP
        FRAME_SKIP = k
        self.frame_count = 0

    def step(self, action, **kwargs):
        r = 0.0
        merge_info = {}
        for k in range(self.frame_skip):
            self.frame_count += 1
            obs, reward, done, info = self.env.step(action, **kwargs)
            r += reward

            for key in info.keys():
                if 'reward' in key:
                    # to assure that we don't igonre other reward
                    # if new reward was added, consider its logic here
                    assert (key == 'shaping_reward') or (key == 'r2_reward')
                    merge_info[key] = merge_info.get(key, 0.0) + info[key]
                else:
                    merge_info[key] = info[key]

            if info['target_changed']:
                #merge_info['shaping_reward'] += info['shaping_reward'] * (self.frame_skip - k - 1)
                logger.warn("[FrameSkip] early break since target was changed")
                break

            if done:
                break
        merge_info['frame_count'] = self.frame_count
        return obs, r, done, merge_info

    def reset(self, **kwargs):
        self.frame_count = 0
        return self.env.reset(**kwargs)


class RewardShaping(gym.Wrapper):
    """ A wrapper for reward shaping, note this wrapper must be the first wrapper """

    def __init__(self, env):
        logger.info("[RewardShaping]type:{}".format(type(env)))

        self.step_count = 0
        self.pre_state_desc = None
        self.last_target_vel = None
        self.last_target_change_step = 0
        self.target_change_times = 0
        gym.Wrapper.__init__(self, env)

    @abc.abstractmethod
    def reward_shaping(self, state_desc, reward, done, action):
        """define your own reward computation function
    Args:
        state_desc(dict): state description for current model
        reward(scalar): generic reward generated by env
        done(bool): generic done flag generated by env
    """
        pass

    def step(self, action, **kwargs):
        self.step_count += 1
        obs, r, done, info = self.env.step(action, **kwargs)
        info = self.reward_shaping(obs, r, done, action)
        if info['target_vel'] > 2.75:
            rate = math.sqrt((2.75**2) / (info['target_vel']**2))
            logger.warn('Changing targets, origin targets: {}'.format(
                obs['target_vel']))
            obs['target_vel'][0] = obs['target_vel'][0] * rate
            obs['target_vel'][2] = obs['target_vel'][2] * rate
            logger.warn('Changing targets, new targets: {}'.format(
                obs['target_vel']))
            info['target_vel'] = 2.75
        if info['target_vel'] < -0.25:
            rate = math.sqrt(((-0.25)**2) / (info['target_vel']**2))
            logger.warn('Changing targets, origin targets: {}'.format(
                obs['target_vel']))
            obs['target_vel'][0] = obs['target_vel'][0] * rate
            obs['target_vel'][2] = obs['target_vel'][2] * rate
            logger.warn('Changing targets, new targets: {}'.format(
                obs['target_vel']))
            info['target_vel'] = -0.25

        delta = 0
        if self.last_target_vel is not None:
            delta = np.absolute(
                np.array(self.last_target_vel) - np.array(obs['target_vel']))
        if (self.last_target_vel is None) or np.all(delta < 1e-5):
            info['target_changed'] = False
        else:
            info['target_changed'] = True
            logger.info("[env_wrapper] target_changed, vx:{}   vz:{}".format(
                obs['target_vel'][0], obs['target_vel'][2]))
            self.last_target_change_step = self.step_count
            self.target_change_times += 1
        info['target_change_times'] = self.target_change_times
        self.last_target_vel = obs['target_vel']

        assert 'shaping_reward' in info
        timeout = False
        if self.step_count >= MAXTIME_LIMIT:
            timeout = True
        if done and not timeout:
            # penalty for falling down
            info['shaping_reward'] += FALL_PENALTY
        info['timeout'] = timeout
        self.pre_state_desc = obs
        return obs, r, done, info

    def reset(self, **kwargs):
        self.step_count = 0
        self.last_target_vel = None
        self.last_target_change_step = 0
        self.target_change_times = 0
        obs = self.env.reset(**kwargs)
        self.pre_state_desc = obs
        return obs


class ForwardReward(RewardShaping):
    """ A reward shaping wraper"""

    def __init__(self, env):
        RewardShaping.__init__(self, env)

    def reward_shaping(self, state_desc, r2_reward, done, action):
        target_vel = math.sqrt(state_desc["target_vel"][0]**2 +
                               state_desc["target_vel"][2]**2)
        if state_desc["target_vel"][0] < 0:
            target_vel = -target_vel

        info = {
            'shaping_reward': r2_reward,
            'target_vel': target_vel,
            'r2_reward': r2_reward,
        }
        return info


class ObsTranformerBase(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.step_fea = MAXTIME_LIMIT
        self.raw_obs = None
        global FRAME_SKIP
        self.frame_skip = int(FRAME_SKIP)

    def get_observation(self, state_desc):
        obs = self._get_observation(state_desc)
        if not isinstance(self, PelvisBasedObs):
            cur_vel_x, cur_vel_z, diff_vel_x, diff_vel_z, diff_vel, diff_theta = calc_vel_diff(
                state_desc)
            obs = np.append(obs, [
                cur_vel_x, cur_vel_z, diff_vel_x, diff_vel_z, diff_vel,
                diff_theta
            ])
        else:
            pass
        return obs

    @abc.abstractmethod
    def _get_observation(self, state_desc):
        pass

    def feature_normalize(self, obs, mean, std, duplicate_id):
        scaler_len = mean.shape[0]
        assert obs.shape[0] >= scaler_len
        obs[:scaler_len] = (obs[:scaler_len] - mean) / std
        final_obs = []
        for i in range(obs.shape[0]):
            if i not in duplicate_id:
                final_obs.append(obs[i])
        return np.array(final_obs)

    def step(self, action, **kwargs):
        obs, r, done, info = self.env.step(action, **kwargs)
        if info['target_changed']:
            # reset step_fea when change target
            self.step_fea = MAXTIME_LIMIT

        self.step_fea -= FRAME_SKIP

        self.raw_obs = copy.deepcopy(obs)
        obs = self.get_observation(obs)
        self.raw_obs['step_count'] = MAXTIME_LIMIT - self.step_fea
        return obs, r, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if obs is None:
            return None
        self.step_fea = MAXTIME_LIMIT
        self.raw_obs = copy.deepcopy(obs)
        obs = self.get_observation(obs)
        self.raw_obs['step_count'] = MAXTIME_LIMIT - self.step_fea
        return obs


class PelvisBasedObs(ObsTranformerBase):
    def __init__(self, env):
        ObsTranformerBase.__init__(self, env)
        data = np.load('./pelvisBasedObs_scaler.npz')
        self.mean, self.std, self.duplicate_id = data['mean'], data[
            'std'], data['duplicate_id']
        self.duplicate_id = self.duplicate_id.astype(np.int32).tolist()

    def get_core_matrix(self, yaw):
        core_matrix = np.zeros(shape=(3, 3))
        core_matrix[0][0] = math.cos(yaw)
        core_matrix[0][2] = -1.0 * math.sin(yaw)
        core_matrix[1][1] = 1
        core_matrix[2][0] = math.sin(yaw)
        core_matrix[2][2] = math.cos(yaw)
        return core_matrix

    def _get_observation(self, state_desc):
        o = OrderedDict()
        for body_part in [
                'pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l',
                'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head'
        ]:
            # position
            o[body_part + '_x'] = state_desc['body_pos'][body_part][0]
            o[body_part + '_y'] = state_desc['body_pos'][body_part][1]
            o[body_part + '_z'] = state_desc['body_pos'][body_part][2]
            # velocity
            o[body_part + '_v_x'] = state_desc["body_vel"][body_part][0]
            o[body_part + '_v_y'] = state_desc["body_vel"][body_part][1]
            o[body_part + '_v_z'] = state_desc["body_vel"][body_part][2]

            o[body_part + '_x_r'] = state_desc["body_pos_rot"][body_part][0]
            o[body_part + '_y_r'] = state_desc["body_pos_rot"][body_part][1]
            o[body_part + '_z_r'] = state_desc["body_pos_rot"][body_part][2]

            o[body_part + '_v_x_r'] = state_desc["body_vel_rot"][body_part][0]
            o[body_part + '_v_y_r'] = state_desc["body_vel_rot"][body_part][1]
            o[body_part + '_v_z_r'] = state_desc["body_vel_rot"][body_part][2]

        for joint in [
                'hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l',
                'back'
        ]:
            if 'hip' not in joint:
                o[joint + '_joint_pos'] = state_desc['joint_pos'][joint][0]
                o[joint + '_joint_vel'] = state_desc['joint_vel'][joint][0]
            else:
                for i in range(3):
                    o[joint + '_joint_pos_' +
                      str(i)] = state_desc['joint_pos'][joint][i]
                    o[joint + '_joint_vel_' +
                      str(i)] = state_desc['joint_vel'][joint][i]

        # In NIPS2017, only use activation
        for muscle in sorted(state_desc["muscles"].keys()):
            activation = state_desc["muscles"][muscle]["activation"]
            if isinstance(activation, float):
                activation = [activation]
            for i, val in enumerate(activation):
                o[muscle + '_activation_' + str(i)] = activation[i]

            fiber_length = state_desc["muscles"][muscle]["fiber_length"]
            if isinstance(fiber_length, float):
                fiber_length = [fiber_length]
            for i, val in enumerate(fiber_length):
                o[muscle + '_fiber_length_' + str(i)] = fiber_length[i]

            fiber_velocity = state_desc["muscles"][muscle]["fiber_velocity"]
            if isinstance(fiber_velocity, float):
                fiber_velocity = [fiber_velocity]
            for i, val in enumerate(fiber_velocity):
                o[muscle + '_fiber_velocity_' + str(i)] = fiber_velocity[i]

        # z axis of mass have some problem now, delete it later
        o['mass_x'] = state_desc["misc"]["mass_center_pos"][0]
        o['mass_y'] = state_desc["misc"]["mass_center_pos"][1]
        o['mass_z'] = state_desc["misc"]["mass_center_pos"][2]

        o['mass_v_x'] = state_desc["misc"]["mass_center_vel"][0]
        o['mass_v_y'] = state_desc["misc"]["mass_center_vel"][1]
        o['mass_v_z'] = state_desc["misc"]["mass_center_vel"][2]
        for key in ['talus_l_y', 'toes_l_y']:
            o['touch_indicator_' + key] = np.clip(0.05 - o[key] * 10 + 0.5, 0.,
                                                  1.)
            o['touch_indicator_2_' + key] = np.clip(0.1 - o[key] * 10 + 0.5,
                                                    0., 1.)

        # Tranformer
        core_matrix = self.get_core_matrix(o['pelvis_y_r'])
        pelvis_pos = np.array([o['pelvis_x'], o['pelvis_y'],
                               o['pelvis_z']]).reshape((3, 1))
        pelvis_vel = np.array(
            [o['pelvis_v_x'], o['pelvis_v_y'], o['pelvis_v_z']]).reshape((3,
                                                                          1))
        for body_part in [
                'mass', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l',
                'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head'
        ]:
            # rotation
            if body_part != 'mass':
                o[body_part + '_y_r'] -= o['pelvis_y_r']
                o[body_part + '_v_y_r'] -= o['pelvis_v_y_r']
            # position/velocity
            global_pos = []
            global_vel = []
            for each in ['_x', '_y', '_z']:
                global_pos.append(o[body_part + each])
                global_vel.append(o[body_part + '_v' + each])
            global_pos = np.array(global_pos).reshape((3, 1))
            global_vel = np.array(global_vel).reshape((3, 1))
            pelvis_rel_pos = core_matrix.dot(global_pos - pelvis_pos)
            w = o['pelvis_v_y_r']
            offset = np.array(
                [-w * pelvis_rel_pos[2], 0, w * pelvis_rel_pos[0]])
            pelvis_rel_vel = core_matrix.dot(global_vel - pelvis_vel) + offset
            for i, each in enumerate(['_x', '_y', '_z']):
                o[body_part + each] = pelvis_rel_pos[i][0]
                o[body_part + '_v' + each] = pelvis_rel_vel[i][0]

        for key in ['pelvis_x', 'pelvis_z', 'pelvis_y_r']:
            del o[key]

        current_v = np.array(state_desc['body_vel']['pelvis']).reshape((3, 1))
        pelvis_current_v = core_matrix.dot(current_v)
        o['pelvis_v_x'] = pelvis_current_v[0]
        o['pelvis_v_z'] = pelvis_current_v[2]

        res = np.array(list(o.values()))
        res = self.feature_normalize(
            res, mean=self.mean, std=self.std, duplicate_id=self.duplicate_id)

        feet_dis = ((o['tibia_l_x'] - o['pros_tibia_r_x'])**2 +
                    (o['tibia_l_z'] - o['pros_tibia_r_z'])**2)**0.5
        res = np.append(res, feet_dis)
        remaining_time = (self.step_fea -
                          (MAXTIME_LIMIT / 2.0)) / (MAXTIME_LIMIT / 2.0) * -1.0
        #logger.info('remaining_time fea: {}'.format(remaining_time))
        res = np.append(res, remaining_time)

        # target driven
        target_v = np.array(state_desc['target_vel']).reshape((3, 1))
        pelvis_target_v = core_matrix.dot(target_v)
        diff_vel_x = pelvis_target_v[0] - pelvis_current_v[0]
        diff_vel_z = pelvis_target_v[2] - pelvis_current_v[2]
        diff_vel = np.sqrt(pelvis_target_v[0] ** 2 + pelvis_target_v[2] ** 2) - \
                   np.sqrt(pelvis_current_v[0] ** 2 + pelvis_current_v[2] ** 2)

        target_vel_x = target_v[0]
        target_vel_z = target_v[2]
        target_theta = math.atan(-1.0 * target_vel_z / target_vel_x)
        current_theta = state_desc['body_pos_rot']['pelvis'][1]
        diff_theta = target_theta - current_theta
        res = np.append(res, [
            diff_vel_x[0] / 3.0, diff_vel_z[0] / 3.0, diff_vel[0] / 3.0,
            diff_theta / (np.pi * 3 / 8)
        ])

        return res


if __name__ == '__main__':
    from osim.env import ProstheticsEnv

    env = ProstheticsEnv(visualize=False)
    env.change_model(model='3D', difficulty=1, prosthetic=True)
    env = ForwardReward(env)
    env = FrameSkip(env, 4)
    env = ActionScale(env)
    env = PelvisBasedObs(env)
    for i in range(64):
        observation = env.reset(project=False)
