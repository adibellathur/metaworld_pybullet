import time
import numpy as np
import pybullet as p
from metaworld_pybullet.sawyer_envs.sawyer_pick_place import SawyerPickPlaceEnv


def main():

    t=0.
    LOC_ACC = 0.01
    prevPose=[0,0,0]
    prevPose1=[0,0,0]
    hasPrevPose = 0
    trailDuration = 15

    env = SawyerPickPlaceEnv()
    obs = env.reset()

    while True:
        # obs = env.step()
        
        # iterate over these 3 goal positions 
        goal_pos = [
            [0.0, 0.7, 0.15, 1.0],
            [0.0, 0.7, 0.04, -1.0],
            [0.0, 0.7, 0.01, 0.6],
            [0.1, 0.8, 0.2, 1.0],
        ]
        for goal in goal_pos:
            while(np.linalg.norm(obs[:3] - goal[:3]) > LOC_ACC):
                t = t + 0.05
                time.sleep(0.05)
                
                xyz_action = env.convert_to_xyz_action(goal[:3])
                gripper_action = np.array([goal[3]])
                action = np.concatenate([xyz_action, gripper_action])

                obs, reward, _, _ = env.step(action)
                tcp = obs[:3]
                ls = p.getLinkState(env.body_id, env.end_effector_index)
                if (hasPrevPose):
                    p.addUserDebugLine(prevPose,tcp,[0,0,1],1,trailDuration)
                    p.addUserDebugLine(prevPose1,ls[0],[1,0,0],1,trailDuration)
                prevPose=tcp
                prevPose1=ls[0]
                hasPrevPose = 1
                print("STEP: {}, REWARD: {}".format(env.curr_path_length, reward))


if __name__ == "__main__":
    main()
