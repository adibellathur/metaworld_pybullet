import time
import numpy as np
import pybullet as p

from metaworld_pybullet.sawyer_envs.sawyer_pick_place import SawyerPickPlaceEnv


def main():

    t=0.
    LOC_ACC = 0.025
    prevPose=[0,0,0]
    prevPose1=[0,0,0]
    hasPrevPose = 0
    trailDuration = 15

    env = SawyerPickPlaceEnv()
    obs = env.reset()

    while True:
        # iterate over these 3 goal positions 
        goal_pos = [
            [-0.4, 0.5, 0.4, 0.0],
            [0.4, 0.5, 0.4, 0.5],
            [0, 0.5, 0.1, 1.0],
        ]
        for goal in goal_pos:
            while(np.linalg.norm(obs[:3] - goal[:3]) > LOC_ACC):
                t = t + 0.005
                time.sleep(0.005)
                
                xyz_action = env.convert_to_xyz_action(goal[:3])
                gripper_action = np.array([goal[3]])
                action = np.concatenate([xyz_action, gripper_action])

                obs = env.step(action)

                ls = p.getLinkState(env.body, env.end_effector_index)
                if (hasPrevPose):
                    p.addUserDebugLine(prevPose,goal[:3],[0,0,0.3],1,trailDuration)
                    p.addUserDebugLine(prevPose1,ls[4],[1,0,0],1,trailDuration)
                prevPose=goal[:3]
                prevPose1=ls[4]
                hasPrevPose = 1

                print("STEP: {}, OBS: {}".format(env.curr_path_length, obs))


if __name__ == "__main__":
    main()
