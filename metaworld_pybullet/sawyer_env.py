import numpy as np
import pybullet as p

from gym.spaces import Box
import reward_utils


class SawyerEnv(object):

    def __init__(self,):
        
        self.body_id = None
        self.table_id = None
        self.obj_id = None

        self.end_effector_index = 19
        self.gripper_joint_indices = [20, 22]
        self.left_gripper_tip_index = 21
        self.right_gripper_tip_index = 23
        self.num_joints = 0
        self.damping_coeff = [0.001] * 23
        self.curr_path_length = 0

        self.goal_pos = None
        self.obj_init_pos = None
        self.obj_init_orientation = None
        self._freeze_rand_vec = False
        self._last_rand_vec = None

        self.init_tcp = None
        self.init_left_pad = None
        self.init_right_pad = None

        self.table_pos = [0,0.60,-.75]
        self.table_orientation = p.getQuaternionFromEuler([0,0,0])

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
        self._partially_observable = False
        self._prev_obs = None

        self.max_path_length = 500
        
        return

    def _load_working_table(self):
        table = p.loadURDF('../data/table/table.urdf', self.table_pos, self.table_orientation)
        p.resetBasePositionAndOrientation(table,self.table_pos, self.table_orientation)
        return table

    def reset(self):
        # if (p.connect(p.SHARED_MEMORY) < 0 and not p.isConnected()):
        p.connect(p.GUI)
        p.loadURDF("plane.urdf",[0,0,-.85])
        self.table_id = self._load_working_table()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        
        self.body_id = p.loadURDF("sawyer_robot/sawyer_description/urdf/sawyer.urdf",[0,-0.1,0])
        p.resetBasePositionAndOrientation(self.body_id,[0,-0.1,0],[0,0,0,1])
        self.num_joints = p.getNumJoints(self.body_id)

        self.reset_model()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        
        self.curr_path_length = 0
        self.all_joint_indices = list(range(p.getNumJoints(self.body_id)))
        self._reset_joints()
        self._set_joint_angles(
            joint_indices=[3, 8, 10, 13,],
            angles=[float(np.pi/2), float(-np.pi / 4.0), float(np.pi / 4.0), float(np.pi / 2.0),],
        )
        self.init_tcp = self.tcp_center
        self.init_left_pad = np.array(
            p.getLinkState(
                self.body_id, 
                self.left_gripper_tip_index, 
            )[0]
        )
        self.init_right_pad = np.array(
            p.getLinkState(
                self.body_id, 
                self.right_gripper_tip_index, 
            )[0]
        )

        p.setRealTimeSimulation(False)
        p.setGravity(0,0,-9.8)
        self._prev_obs = self._get_curr_obs_combined_no_goal()
        return self._get_obs()
    
    def _set_joint_angles(self, joint_indices, angles, velocities=0):
        for i, (j, a) in enumerate(zip(joint_indices, angles)):
            p.resetJointState(
                self.body_id,
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
    
    def step(self, action=None):
        if action is not None and action.any():
            curr_pos = self._prev_obs[:3]
            action = np.clip(action, -1.0, 1.0)
            delta_pos = action[:3]
            gripper_pos = action[3]
            gripper_pos = [-0.5 * gripper_pos, 0.5 * gripper_pos]

            joint_poses = p.calculateInverseKinematics(
                self.body_id,
                self.end_effector_index,
                targetPosition=curr_pos + delta_pos,
                targetOrientation=p.getQuaternionFromEuler([0, -np.pi, 0]),
                jointDamping=self.damping_coeff
            )
            for i in range (self.num_joints):
                jointInfo = p.getJointInfo(self.body_id, i)
                qIndex = jointInfo[3]
                if qIndex > -1:
                    p.setJointMotorControlArray(
                        bodyIndex=self.body_id,
                        controlMode=p.POSITION_CONTROL,
                        jointIndices=[i,],
                        targetPositions=[joint_poses[qIndex-7],],
                    )
            p.setJointMotorControlArray(
                self.body_id, 
                jointIndices=self.gripper_joint_indices, 
                controlMode=p.VELOCITY_CONTROL, 
                targetVelocities=gripper_pos, 
                forces=[100.0, 100.0]
            )
        p.stepSimulation()
        self.curr_path_length += 1
        obs = self._get_obs()
        reward, info = self.evaluate_state(action, obs)
        return obs, reward, False, info

    def _get_curr_obs_combined_no_goal(self,):
        hand_pos = self.tcp_center
        left_gripper_pos = np.array(
            p.getLinkState(
                self.body_id, 
                self.left_gripper_tip_index, 
            )[0]
        )
        right_gripper_pos = np.array(
            p.getLinkState(
                self.body_id, 
                self.right_gripper_tip_index, 
            )[0]
        )
        gripper_dist = np.array([
            np.clip(
                np.linalg.norm(left_gripper_pos - right_gripper_pos) / 0.1,
                0.0,
                1.0,
            )
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
                self.body_id, 
                self.left_gripper_tip_index, 
            )[0]
        )
        right_gripper_pos = np.array(
            p.getLinkState(
                self.body_id, 
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
        return  self.goal_pos

    def reset_model(self,):
        '''
        to be replaced in all child classes
        '''
        return
    
    def evaluate_state(self, obs, action):
        '''
        to be replaced in all child classes
        '''
        return 0, {}

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
    
    def _get_random_pos(self, obj_init_space, goal_space):
        if self._freeze_rand_vec:
            return self.obj_init_pos, self.goal_pos
        else:
            obj_rand_vec = np.random.uniform(
                obj_init_space.low,
                obj_init_space.high,
                size=obj_init_space.low.size,
            )
            goal_rand_vec = np.random.uniform(
                goal_space.low,
                goal_space.high,
                size=goal_space.low.size,
            )
            return obj_rand_vec, goal_rand_vec
        
    def _gripper_caging_reward(self,
                               action,
                               obj_pos,
                               obj_radius,
                               pad_success_thresh,
                               object_reach_radius,
                               xz_thresh,
                               desired_gripper_effort=1.0,
                               high_density=False,
                               medium_density=False):
        """Reward for agent grasping obj
            Args:
                action(np.ndarray): (4,) array representing the action
                    delta(x), delta(y), delta(z), gripper_effort
                obj_pos(np.ndarray): (3,) array representing the obj x,y,z
                obj_radius(float):radius of object's bounding sphere
                pad_success_thresh(float): successful distance of gripper_pad
                    to object
                object_reach_radius(float): successful distance of gripper center
                    to the object.
                xz_thresh(float): successful distance of gripper in x_z axis to the
                    object. Y axis not included since the caging function handles
                        successful grasping in the Y axis.
        """
        if high_density and medium_density:
            raise ValueError("Can only be either high_density or medium_density")
        # MARK: Left-right gripper information for caging reward----------------
        left_pad = np.array(
            p.getLinkState(
                self.body_id, 
                self.left_gripper_tip_index, 
            )[0]
        )
        right_pad = np.array(
            p.getLinkState(
                self.body_id, 
                self.right_gripper_tip_index, 
            )[0]
        )

        # get current positions of left and right pads (Y axis)
        pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
        # compare *current* pad positions with *current* obj position (Y axis)
        pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
        # compare *current* pad positions with *initial* obj position (Y axis)
        pad_to_objinit_lr = np.abs(pad_y_lr - self.obj_init_pos[1])

        # Compute the left/right caging rewards. This is crucial for success,
        # yet counterintuitive mathematically because we invented it
        # accidentally.
        #
        # Before touching the object, `pad_to_obj_lr` ("x") is always separated
        # from `caging_lr_margin` ("the margin") by some small number,
        # `pad_success_thresh`.
        #
        # When far away from the object:
        #       x = margin + pad_success_thresh
        #       --> Thus x is outside the margin, yielding very small reward.
        #           Here, any variation in the reward is due to the fact that
        #           the margin itself is shifting.
        # When near the object (within pad_success_thresh):
        #       x = pad_success_thresh - margin
        #       --> Thus x is well within the margin. As long as x > obj_radius,
        #           it will also be within the bounds, yielding maximum reward.
        #           Here, any variation in the reward is due to the gripper
        #           moving *too close* to the object (i.e, blowing past the
        #           obj_radius bound).
        #
        # Therefore, before touching the object, this is very nearly a binary
        # reward -- if the gripper is between obj_radius and pad_success_thresh,
        # it gets maximum reward. Otherwise, the reward very quickly falls off.
        #
        # After grasping the object and moving it away from initial position,
        # x remains (mostly) constant while the margin grows considerably. This
        # penalizes the agent if it moves *back* toward `obj_init_pos`, but
        # offers no encouragement for leaving that position in the first place.
        # That part is left to the reward functions of individual environments.
        caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
        caging_lr = [reward_utils.tolerance(
            pad_to_obj_lr[i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[i],  # "margin" in the description above
            sigmoid='long_tail',
        ) for i in range(2)]
        caging_y = reward_utils.hamacher_product(*caging_lr)

        # MARK: X-Z gripper information for caging reward-----------------------
        tcp = self.tcp_center
        xz = [0, 2]

        # Compared to the caging_y reward, caging_xz is simple. The margin is
        # constant (something in the 0.3 to 0.5 range) and x shrinks as the
        # gripper moves towards the object. After picking up the object, the
        # reward is maximized and changes very little
        caging_xz_margin = np.linalg.norm(self.obj_init_pos[xz] - self.init_tcp[xz])
        caging_xz_margin -= xz_thresh
        caging_xz = reward_utils.tolerance(
            np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
            bounds=(0, xz_thresh),
            margin=caging_xz_margin,  # "margin" in the description above
            sigmoid='long_tail',
        )

        # MARK: Closed-extent gripper information for caging reward-------------
        gripper_closed = min(max(0, action[-1]), desired_gripper_effort) \
                         / desired_gripper_effort

        # MARK: Combine components----------------------------------------------
        caging = reward_utils.hamacher_product(caging_y, caging_xz)
        gripping = gripper_closed if caging > 0.97 else 0.
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2
        if medium_density:
            tcp = self.tcp_center
            tcp_to_obj = np.linalg.norm(obj_pos - tcp)
            tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
            # Compute reach reward
            # - We subtract `object_reach_radius` from the margin so that the
            #   reward always starts with a value of 0.1
            reach_margin = abs(tcp_to_obj_init - object_reach_radius)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, object_reach_radius),
                margin=reach_margin,
                sigmoid='long_tail',
            )
            caging_and_gripping = (caging_and_gripping + reach) / 2
        return caging_and_gripping

    def convert_to_xyz_action(self, desired_pos, scale=20):
        curr_pos = self._get_obs()[:3]
        delta_pos = (desired_pos - curr_pos) /(scale *  np.linalg.norm(desired_pos - curr_pos))
        return np.array(delta_pos)

    def convert_to_gripper_action(self, gripper_distance):
        return (gripper_distance / 0.02)

    def convert_to_gripper_distance(self, gripper_action):
        return gripper_action * 0.02
