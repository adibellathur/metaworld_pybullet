import time

import numpy as np
import pybullet as p

from gym.spaces import Box


class SawyerEnv(object):

	def __init__(self,):
		self.damping_coeff = [0.001] * 23
		self.body = None
		self.end_effector_index = 19
		self.real_time_simulation = False
		self.num_joints = 0
		self.table = None

		self.hand_low = None
		self.hand_high = None
		self.curr_path_length = 0

		self.action_space = Box(
			np.array([-1, -1, -1, -1]),
			np.array([+1, +1, +1, +1]),
		)
		self.all_joint_indices = None

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
		clid = p.connect(p.SHARED_MEMORY)
		if (clid < 0):
			p.connect(p.GUI)
		p.loadURDF("plane.urdf",[0,0,-.98])
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
		self.body = p.loadURDF("sawyer_robot/sawyer_description/urdf/sawyer.urdf",[0,0,0])
		self.table = self.load_ugly_table()
		self.num_joints = p.getNumJoints(self.body)
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
		p.resetBasePositionAndOrientation(self.body,[0,0,0],[0,0,0,1])

		self.all_joint_indices = list(range(p.getNumJoints(self.body)))
		print(self.all_joint_indices)

		self.reset_joints()

		self.set_joint_angles(
			joint_indices=[8,10,13,],
			angles=[float(-np.pi / 4.0), float(np.pi / 4.0), float(np.pi / 2.0),],
		)
		p.setRealTimeSimulation(self.real_time_simulation)
		p.setGravity(0,0,0)
		return

	def step(self, desired_pos):
		joint_poses = p.calculateInverseKinematics(
			self.body,
			self.end_effector_index,
			desired_pos,
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
		p.stepSimulation()
		return

	def get_obs():
		'''
		TODO: SEBASTIAN
		Here is where you need to calculate the final position of the arm, and return the [x,y,z] location 
		as an array. The index of the end of the arm is saved as self.end_effector_index, and as a start
		I suggest using p.forwardKinematics() or something similar. I think you should start by getting 
		all the joint positions, and then passing them into the forward kinematics function to get the final
		end effector position.
		
		the best place for finding out about the pybullet api is here: 
		https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html
		or here:
		https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstartguide.pdf
		'''
		pass

	def set_joint_angles(self, joint_indices, angles, use_limits=True, velocities=0):
		for i, (j, a) in enumerate(zip(joint_indices, angles)):
			p.resetJointState(
				self.body,
				jointIndex=j,
				# targetValue=min(max(a, self.lower_limits[j]), self.upper_limits[j]) if use_limits else a, 
				targetValue=a, 
				targetVelocity=velocities if type(velocities) in [int, float] else velocities[i]
			)

	def reset_joints(self):
		self.set_joint_angles(
			self.all_joint_indices, 
			[0.]*len(self.all_joint_indices)
		)



def main():
	t=0.
	prevPose=[0,0,0]
	prevPose1=[0,0,0]
	hasPrevPose = 0
	trailDuration = 15

	env = SawyerEnv()
	env.reset()
		
	while True:
		for i in range (20):
			t=t+0.01
			time.sleep(0.01)
			pos = [0.5, 0.25, 0.5]
			env.step(pos)

			ls = p.getLinkState(env.body, env.end_effector_index)
			if (hasPrevPose):
				p.addUserDebugLine(prevPose,pos,[0,0,0.3],1,trailDuration)
				p.addUserDebugLine(prevPose1,ls[4],[1,0,0],1,trailDuration)
			prevPose=pos
			prevPose1=ls[4]
			hasPrevPose = 1

			'''
			TODO: SEBASTIAN
			if your get_obs() function works, you should be able to print the position of 
			your arm at every step, and watch as it gets closer to [0.5, 0.25, 0.5] (pos above).
			'''
			# print(env.get_obs())

		for i in range (20):
			t=t+0.01
			time.sleep(0.01)
			pos = [1.0, 0.0, 0.0]
			env.step(pos)

			ls = p.getLinkState(env.body, env.end_effector_index)
			if (hasPrevPose):
				p.addUserDebugLine(prevPose,pos,[0,0,0.3],1,trailDuration)
				p.addUserDebugLine(prevPose1,ls[4],[1,0,0],1,trailDuration)
			prevPose=pos
			prevPose1=ls[4]
			hasPrevPose = 1
			'''
			TODO: SEBASTIAN
			if your get_obs() function works, you should be able to print the position of 
			your arm at every step, and watch as it gets closer to [1.0, 0.0, 0.0] (pos above).
			'''
			# print(env.get_obs())

		for i in range (20):
			t=t+0.01
			time.sleep(0.01)
			pos = [0.75, -0.2, 0.5]
			env.step(pos)

			ls = p.getLinkState(env.body, env.end_effector_index)
			if (hasPrevPose):
				p.addUserDebugLine(prevPose,pos,[0,0,0.3],1,trailDuration)
				p.addUserDebugLine(prevPose1,ls[4],[1,0,0],1,trailDuration)
			prevPose=pos
			prevPose1=ls[4]
			hasPrevPose = 1	
			'''
			TODO: SEBASTIAN
			if your get_obs() function works, you should be able to print the position of 
			your arm at every step, and watch as it gets closer to [0.75, -0.2, 0.5] (pos above).
			'''
			# print(env.get_obs())

if __name__ == "__main__":
	main()
