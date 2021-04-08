import numpy as np
import pybullet as p

from gym.spaces import Box


class SawyerEnv(object):

	def __init__(self,):
		
		self.body = None
		self.table = None

		self.end_effector_index = 19
		self.gripper_indices = [20, 22]
		self.num_joints = 0
		self.damping_coeff = [0.001] * 23
		self.curr_path_length = 0
		self.real_time_simulation = False

		self.hand_low = None
		self.hand_high = None
		
		self.action_space = Box(
			np.array([-1, -1, -1, -1]),
			np.array([+1, +1, +1, +1]),
		)
		self.all_joint_indices = None
		self.prev_pos = None

		return

	def load_ugly_table(self):
		return p.loadMJCF('data/basic_scene_test.xml')


	def load_table(self):
		tablebody_collision_id = p.createCollisionShape(
    		shapeType=p.GEOM_MESH,
    		fileName="data/tablebody.stl",
			flags=p.URDF_USE_SELF_COLLISION
		)
		tablebody_visual_id = p.createVisualShape(
			shapeType=p.GEOM_MESH,
			fileName="data/tablebody.stl",
		)
		tabletop_collision_id = p.createCollisionShape(
    		shapeType=p.GEOM_MESH,
    		fileName="data/tabletop.stl",
			flags=p.URDF_USE_SELF_COLLISION
		)

		tabletop_visual_id = p.createVisualShape(
			shapeType=p.GEOM_MESH,
			fileName="data/tabletop.stl",
		)
		body_id = p.createMultiBody(
			#baseMass=0,
			#baseCollisionShapeIndex=tablebody_collision_id,
			#baseVisualShapeIndex=tablebody_visual_id,
			linkMasses=[0,0],
			linkCollisionShapeIndices=[tablebody_collision_id, tabletop_collision_id],
			linkVisualShapeIndices=[tablebody_visual_id, tabletop_visual_id],
			linkPositions=[[0,0,0],[0,0,0]],
			linkOrientations=[[0, 0, 0, 1],[0, 0, 0, 1]],
		)
		return body_id

	def reset(self):
		if (p.connect(p.SHARED_MEMORY) < 0):
			p.connect(p.GUI)
		p.loadURDF("plane.urdf",[0,0,-.98])

		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

		self.table = self.load_ugly_table()
		self.body = p.loadURDF("sawyer_robot/sawyer_description/urdf/sawyer.urdf",[0,0,0])
		p.resetBasePositionAndOrientation(self.body,[0,0,0],[0,0,0,1])
		self.num_joints = p.getNumJoints(self.body)

		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
		
		self.all_joint_indices = list(range(p.getNumJoints(self.body)))
		self._reset_joints()
		self._set_joint_angles(
			joint_indices=[3, 8, 10, 13,],
			angles=[float(np.pi/2), float(-np.pi / 4.0), float(np.pi / 4.0), float(np.pi / 2.0),],
		)

		p.setRealTimeSimulation(self.real_time_simulation)
		p.setGravity(0,0,-9.8)
		return self._get_obs()
	
	def convert_to_xyz_action(self, desired_pos):
		curr_pos = self._get_obs()[:3]
		delta_pos = (desired_pos - curr_pos) /(10 *  np.linalg.norm(desired_pos - curr_pos))
		return np.array(delta_pos)

	def convert_to_gripper_action(self, gripper_distance):
		return (gripper_distance / 0.02)
	
	def convert_to_gripper_distance(self, gripper_action):
		return gripper_action * 0.02

	def step(self, action=None):
		if action.any():
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
				jointIndices=self.gripper_indices, 
				controlMode=p.POSITION_CONTROL, 
				targetPositions=gripper_pos, 
			)
		p.stepSimulation()
		self.curr_path_length += 1
		return self._get_obs()

	def _get_obs(self,):
		link_position = np.array(
			p.getLinkState(
				self.body, 
				self.end_effector_index, 
				computeForwardKinematics=True
			)[0]
		)
		gripper_pos = np.array([
			p.getJointState(
				self.body, 
				self.gripper_indices[0]
			)[0]
		])
		obs = np.concatenate([
			link_position,
			gripper_pos,
			self._get_obj_pos(),
			self._get_obj_orientation()
		])
		return obs
	
	def _get_obj_pos(self,):
		'''
		to be replaced in all child classes
		'''
		return np.array([0,0,0,])

	def _get_obj_orientation(self,):
		'''
		to be replaced in all child classes
		'''
		return np.array([0,0,0,0,])

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