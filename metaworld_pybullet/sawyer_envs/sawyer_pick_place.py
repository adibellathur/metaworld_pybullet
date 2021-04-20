import numpy as np
import pybullet as p

from gym.spaces import Box
from metaworld_pybullet import reward_utils
from metaworld_pybullet.sawyer_env import SawyerEnv


class SawyerPickPlaceEnv(SawyerEnv):

    def __init__(self):
        super().__init__()
        self.obj_id = None

        self.goal = np.array([0.0, 0.6, 0.2])
        self.obj_init_pos = np.array([0,.7,.0])
        self.obj_init_orientation = p.getQuaternionFromEuler([0,0,0])

        self._random_reset_space = Box(
            np.hstack(([-0.1, 0.6, 0.0], [-0.1, 0.8, 0.05])),
            np.hstack(([0.1, 0.7,0.0], [0.1, 0.9, 0.3])),
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
        # obs = super().reset()
        self.obj_id = self._load_obj()
        # TODO: OBJECT RANDOMIZATION
        return

    def evaluate_state(self, obs, action):
        obj = obs[4:7]
        reward, tcp_to_obj, tcp_open, obj_to_target, grasp_reward, in_place_reward = self.compute_reward(action, obs)
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = 0
        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
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
        target = self._target_pos

        obj_to_target = np.linalg.norm(obj - target)
        tcp_to_obj = np.linalg.norm(obj - tcp)
        in_place_margin = (np.linalg.norm(self.obj_init_pos - target))

        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)

        object_grasped = self._gripper_caging_reward(action, obj)
        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
                                                                    in_place)
        reward = in_place_and_object_grasped

        if tcp_to_obj < 0.02 and (tcp_opened > 0) and (obj[2] - 0.01 > self.obj_init_pos[2]):
            reward += 1. + 5. * in_place
        if obj_to_target < _TARGET_RADIUS:
            reward = 10.
        return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place]
