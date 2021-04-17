import numpy as np
import pybullet as p

from gym.spaces import Box
from metaworld_pybullet.sawyer_env import SawyerEnv


class SawyerPickPlaceEnv(SawyerEnv):

    def __init__(self):
        super().__init__()
        self.goal = np.array([0.0, 0.6, 0.2])
        self.puck = None
        return

    
    def _load_puck(self):
        startPos = [0,.7,.2]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        return p.loadURDF('../data/table/puck.urdf', startPos, startOrientation)
        

    def reset(self):
        obs = super().reset()
        self.puck = self._load_puck()
        # TODO: SEBASTIAN - ADD NEW OBJECTS HERE TO THIS RESET FUNCTION
        return obs
    
    def _get_obj_pos(self,):

        return np.array([0,0,0,])

    def _get_obj_orientation(self,):
        return np.array([0,0,0,0,])
    
    def compute_reward(action, obs):
        return 0.0
