import numpy as np
import pybullet as p

from gym.spaces import Box


class SawyerEnv(object):

    def __init__(self,):
        
        self.body = None
        self.table = None

        self.end_effector_index = 19
        self.gripper_joint_indices = [20, 22]
        self.left_gripper_tip_index = 21
        self.right_gripper_tip_index = 23
        self.num_joints = 0
        self.damping_coeff = [0.001] * 23
        self.curr_path_length = 0
        self.real_time_simulation = False

        self._goal_pos = None
        self.obj_init_pos = None
        self.obj_init_orientation = None

        self.hand_space = Box(
            np.array([-0.525, .348, -.0525]),
            np.array([+0.525, 1.025, .7]),
        )
        self.gripper_space = Box(
            np.array([-1.]),
            np.array([+1.]),
        )
        self.obj_space = Box(
            np.full(14, -np.inf),
            np.full(14, +np.inf),
        )
        self.goal_space = Box(
            np.zeros(3),
            np.zeros(3),
        )
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
        )
        self.all_joint_indices = None
        self.prev_pos = None
        self._partially_observable = False
        self._prev_obs = None
        return

    def load_working_table(self):
        startPos = [0,1,-.8]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        return p.loadURDF('../data/table/table.urdf', startPos, startOrientation)

    def reset(self):
        if (p.connect(p.SHARED_MEMORY) < 0):
            p.connect(p.GUI)
        p.loadURDF("plane.urdf",[0,0,-.98])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

        self.table = self.load_working_table()
        self.body = p.loadURDF("sawyer_robot/sawyer_description/urdf/sawyer.urdf",[0,0,0])
        p.resetBasePositionAndOrientation(self.body,[0,0,0],[0,0,0,1])
        self.num_joints = p.getNumJoints(self.body)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        
        self.curr_path_length = 0
        self.all_joint_indices = list(range(p.getNumJoints(self.body)))
        self._reset_joints()
        self._set_joint_angles(
            joint_indices=[3, 8, 10, 13,],
            angles=[float(np.pi/2), float(-np.pi / 4.0), float(np.pi / 4.0), float(np.pi / 2.0),],
        )

        p.setRealTimeSimulation(self.real_time_simulation)
        p.setGravity(0,0,-9.8)
        self._prev_obs = self._get_curr_obs_combined_no_goal()
        return self._get_obs()
    
    def step(self, action=None):
        if action is not None and action.any():
            curr_pos = self._get_obs()[:3]
            action = np.clip(action, -1.0, 1.0)
            delta_pos = action[:3]
            gripper_pos = self.convert_to_gripper_distance(action[3])
            gripper_pos = [gripper_pos, -gripper_pos]

            joint_poses = p.calculateInverseKinematics(
                self.body,
                self.end_effector_index,
                targetPosition=curr_pos + delta_pos,
                targetOrientation=p.getQuaternionFromEuler([0, -np.pi, 0]),
                jointDamping=self.damping_coeff
            )
            for i in range (self.num_joints):
                jointInfo = p.getJointInfo(self.body, i)
                qIndex = jointInfo[3]
                if qIndex > -1:
                    p.setJointMotorControlArray(
                        bodyIndex=self.body,
                        controlMode=p.POSITION_CONTROL,
                        jointIndices=[i,],
                        targetPositions=[joint_poses[qIndex-7],],
                    )
            p.setJointMotorControlArray(
                self.body, 
                jointIndices=self.gripper_joint_indices, 
                controlMode=p.POSITION_CONTROL, 
                targetPositions=gripper_pos, 
            )
        p.stepSimulation()
        self.curr_path_length += 1
        return self._get_obs()

    def _get_curr_obs_combined_no_goal(self,):
        # hand_pos = np.array(
        #     p.getLinkState(
        #         self.body, 
        #         self.end_effector_index, 
        #         computeForwardKinematics=True
        #     )[0]
        # )
        hand_pos = self.tcp_center
        gripper_dist = np.array([
            p.getJointState(
                self.body, 
                self.gripper_joint_indices[0]
            )[0]
        ])
        obj_pos = self._get_obj_pos()
        obj_orientation = self._get_obj_orientation()
        obs = np.concatenate([
            hand_pos,
            gripper_dist,
            obj_pos[:3],
            obj_orientation[:4],
            obj_pos[3:],
            obj_orientation[4:],
        ])
        return obs
    
    def _get_obs(self,):
        curr_obs = self._get_curr_obs_combined_no_goal()
        goal_pos = self._get_goal_pos()
        if self._partially_observable:
            goal_pos = np.zeros_like(goal_pos)
            
        obs = np.concatenate([
            curr_obs,
            self._prev_obs,
            goal_pos,
        ])
        self._prev_obs = curr_obs
        return obs
    
    @property
    def tcp_center(self,):
        left_gripper_pos = np.array(
            p.getLinkState(
                self.body, 
                self.left_gripper_tip_index, 
            )[0]
        )
        right_gripper_pos = np.array(
            p.getLinkState(
                self.body, 
                self.right_gripper_tip_index, 
            )[0]
        )
        tcp_center = np.average([left_gripper_pos, right_gripper_pos], axis=0)
        return tcp_center
    
    def _get_obj_pos(self,):
        '''
        to be replaced in all child classes
        '''
        return np.array([0.,0.,0.,0.,0.,0.,])

    def _get_obj_orientation(self,):
        '''
        to be replaced in all child classes
        '''
        return np.array([0.,0.,0.,0.,0.,0.,0.,0.,])
    
    def _get_goal_pos(self,):
        '''
        to be replaced in all child classes
        '''
        return  np.array([0.,0.,0.])
    
    @property
    def observation_space(self):
        goal_low = self.goal_space.low
        goal_high = self.goal_space.high
        if self._partially_observable:
            goal_low = np.zeros(3)
            goal_high = np.zeros(3)

        return Box(
            np.concatenate([
                self.hand_space.low, 
                self.gripper_space.low, 
                self.obj_space.low,
                self.hand_space.low,
                self.gripper_space.low,
                self.obj_space.low,
                goal_low,
            ]),
            np.concatenate([
                self.hand_space.high,
                self.gripper_space.high,
                self.obj_space.high,
                self.hand_space.high,
                self.gripper_space.high,
                self.obj_space.high,
                goal_high,
            ])
        )
    
    def convert_to_xyz_action(self, desired_pos, scale=10):
        curr_pos = self._get_obs()[:3]
        delta_pos = (desired_pos - curr_pos) /(scale *  np.linalg.norm(desired_pos - curr_pos))
        return np.array(delta_pos)

    def convert_to_gripper_action(self, gripper_distance):
        return (gripper_distance / 0.02)
    
    def convert_to_gripper_distance(self, gripper_action):
        return gripper_action * 0.02

    def _set_joint_angles(self, joint_indices, angles, velocities=0):
        for i, (j, a) in enumerate(zip(joint_indices, angles)):
            p.resetJointState(
                self.body,
                jointIndex=j,
                # targetValue=min(max(a, self.lower_limits[j]), self.upper_limits[j]) if use_limits else a, 
                targetValue=a, 
                targetVelocity=velocities if type(velocities) in [int, float] else velocities[i]
            )
        return

    def _reset_joints(self):
        self._set_joint_angles(
            self.all_joint_indices, 
            [0.]*len(self.all_joint_indices)
        )
        return
