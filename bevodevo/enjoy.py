import os
import sys
import argparse
import subprocess

import torch
import numpy as np
import time

import gym
import pybullet
import pybullet_envs
import matplotlib.pyplot as plt


from mpi4py import MPI
comm = MPI.COMM_WORLD

#from bevodevo.policies.rnns import GatedRNNPolicy
from bevodevo.policies.cnns import ImpalaCNNPolicy
from bevodevo.policies.mlps import MLPPolicy

from bevodevo.algos.es import ESPopulation
from bevodevo.algos.cmaes import CMAESPopulation
from bevodevo.algos.pges import PGESPopulation
from bevodevo.algos.ga import GeneticPopulation
from bevodevo.algos.random_search import RandomSearch

#from bevodevo.algos.dqn import DQN

def enjoy(argv):

    # env_name, generations, population_size, 
    

    if "elite_pop" not in argv.file_path and ".pt" not in argv.file_path:
        my_dir = os.listdir(argv.file_path)
        latest_gen = 0
        my_file_path = ""
        for filename in my_dir:
            if "elite_pop" in filename:
                current_gen = int(filename.split("_")[5]) 
                if latest_gen < current_gen: 
                    latest_gen = current_gen
                    my_file_path = argv.file_path + filename
    else:
        my_file_path = argv.file_path

    print(my_file_path)

    if "gatedrnn" in argv.policy.lower():
        policy_fn = SimpleGatedRNNPolicy
    elif "impala"  in argv.policy.lower():
        policy_fn = ImpalaCNNPolicy
    elif "mlppolicy" in argv.policy.lower():
        policy_fn = MLPPolicy
    else:
        print("policy not found, resorting to default MLP policy")
        policy_fn = MLPPolicy

    if ".npy" in my_file_path:
        my_data = np.load(my_file_path, allow_pickle=True)[np.newaxis][0]
        env_name = my_data["env_name"]
    else:
        env_name = argv.env_name

    env = gym.make(env_name)

    if argv.no_render:
        gym_render = False
    else:
        if "BulletEnv" in env_name:
            env.render()
            gym_render = False
        else:
            gym_render = True

    obs_dim = env.observation_space.shape

    if len(obs_dim) == 3:
        obs_dim = obs_dim
    else:
        obs_dim = obs_dim[0]
    hid_dim = [32,32] 


    try:
        act_dim = env.action_space.n
        discrete = True
    except:
        act_dim = env.action_space.sample().shape[0]
        discrete = False

    no_array = act_dim == 2 and discrete 

    if ".npy" in my_file_path:
        parameters = np.load(my_file_path, allow_pickle=True)[np.newaxis][0]
    else:
        parameters = torch.load(my_file_path)

    if type(parameters) is dict:
        elite_keep = len(parameters)
    else:
        elite_keep = 1

    for agent_idx in range(min([argv.num_agents, elite_keep])):

        if type(parameters) is dict:
            agent_args = {"dim_x": obs_dim, "dim_h": hid_dim, \
                    "dim_y": act_dim, "params": parameters["elite_0"]} 
        else:
            agent_args = {"dim_x": obs_dim, "dim_h": hid_dim, \
                    "dim_y": act_dim, "params": parameters} 
            if ".pt" in my_file_path:
                agent_args["params"] = None

        agent = policy_fn(agent_args, discrete=discrete)

        if ".pt" in my_file_path:
            agent.load_state_dict(parameters)
            agent_args["params"] = agent.get_params()

        agent.set_params(agent_args["params"])

        epd_rewards = []
        for episode in range(argv.episodes):
            obs = env.reset()
            sum_reward = 0.0
            done = False
            step_count = 0
            while not done:

                action = agent.get_action(obs)

                if no_array and type(action) == np.ndarray\
                        or len(action.shape) > 1:

                    action = action[0]

                obs, reward, done, info = env.step(action)
                step_count += 1

                sum_reward += reward

                if gym_render:
                    env.render()
                    time.sleep(1e-2)
                if argv.save_frames:
                    
                    if "BulletEnv" in argv.env_name:
                        env.unwrapped._render_width = 640
                        env.unwrapped._render_height = 480

                    img = env.render(mode="rgb_array")
                    plt.figure()
                    plt.imshow(img)
                    plt.savefig("./frames/frame_agent{}_pd{}_step{}.png".format(\
                            agent_idx, episode, step_count))
                    plt.close()

                time.sleep(0.01)
                if step_count >= argv.max_steps:
                    done = True




            epd_rewards.append(sum_reward)

        print("reward stats for elite {} over {} epds:".format(agent_idx, argv.episodes))
        print("mean rew: {:.3e}, +/- {:.3e} std. dev.".format(np.mean(epd_rewards), np.std(epd_rewards)))
        print("max rew: {:.3e}, min rew: {:.3e}".format(np.max(epd_rewards), np.min(epd_rewards)))

    env.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Experiment parameters")

    parser.add_argument("-n", "--env_name", type=str, \
            help="name of environemt", default="InvertedPendulumBulletEnv-v0")
    parser.add_argument("-pi", "--policy", type=str,\
            help="name of policy architecture", default="MLPPolicy")
    parser.add_argument("-e", "--episodes", type=int,\
            help="number of episodes", default=5)
    parser.add_argument("-s", "--save_frames", type=bool, \
            help="save frames or not (not implemented)", default=False)
    parser.add_argument("-nr", "--no_render", type=bool,\
            help="don't render", default=False)
    parser.add_argument("-ms", "--max_steps", type=int,\
            help="maximum number of steps per episode", default=4000)
    parser.add_argument("-f", "--file_path", type=str,\
            help="file path to model parameters", \
            default="./results/test_exp/")
    parser.add_argument("-a", "--num_agents", type=int,\
            help="how many agents to evaluate", \
            default=1)



    args = parser.parse_args()

    enjoy(args)
