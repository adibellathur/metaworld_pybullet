import numpy as np
import pybullet as p

from gym.spaces import Box
from metaworld_pybullet import reward_utils
from metaworld_pybullet.sawyer_env import SawyerEnv


class SawyerPickPlaceEnv(SawyerEnv):

    def __init__(self):
        super().__init__()
        self.obj_id = None

        self.obj_init_pos = np.array([0,.7,.0])
        self.obj_init_orientation = p.getQuaternionFromEuler([0,0,0])
        self.goal_pos = np.array([0.1, 0.8, 0.2])

        self.obj_init_space = Box(
            np.array([-0.1, 0.6, 0.0]),
            np.array([0.1, 0.7,0.0]),
        )

        self.goal_space = Box(
            np.array([-0.1, 0.8, 0.05]), 
            np.array([0.1, 0.9, 0.3])
        )
        return

    def _load_obj(self):
        obj_id = p.loadURDF('../data/table/puck.urdf', self.obj_init_pos, self.obj_init_orientation)
        p.resetBasePositionAndOrientation(obj_id, self.obj_init_pos, self.obj_init_orientation)
        return obj_id

    def reset_model(self):
        self.obj_init_pos, self.goal_pos = self._get_random_pos(
            self.obj_init_space, 
            self.goal_space
        )
        while(np.linalg.norm(self.obj_init_pos[:2] - self.goal_pos[:2]) < 0.15):
            self.obj_init_pos, self.goal_pos = self._get_random_pos(
                self.obj_init_space,
                self.goal_space
            )
        self.obj_id = self._load_obj()
        return
    
    def _get_obj_pos(self,):
        obj_pos = np.array(
            p.getBasePositionAndOrientation(
                self.obj_id,
            )[0]
        )
        return np.concatenate([obj_pos, np.array([0., 0., 0.,])])

    def _get_obj_orientation(self,):
        obj_ori = np.array(
            p.getBasePositionAndOrientation(
                self.obj_id,
            )[1]
        )
        return np.concatenate([obj_ori, np.array([0., 0., 0., 0.,])])

    def evaluate_state(self, action, obs):
        obj = obs[4:7]
        reward, tcp_to_obj, tcp_open, obj_to_target, grasp_reward, in_place_reward = self.compute_reward(action, obs)
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        info = {
            'success': success,
            'near_object': near_object,
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward
        }
        return reward, info
    
    def compute_reward(self, action, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = self.goal_pos

        obj_to_target = np.linalg.norm(obj - target)
        tcp_to_obj = np.linalg.norm(obj - tcp)
        in_place_margin = (np.linalg.norm(self.obj_init_pos - target))

        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.05,
            xz_thresh=0.005,
            high_density=True
        )
        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
                                                                    in_place)
        reward = in_place_and_object_grasped
        if tcp_to_obj < 0.02 and (tcp_opened > 0) and (obj[2] - 0.01 > self.obj_init_pos[2]):
            reward += 1. + 5. * in_place
        if obj_to_target < _TARGET_RADIUS:
            reward = 10.
        return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place]
