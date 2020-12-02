import os
import sys
import subprocess

import torch
import torch.nn.functional as F
import numpy as np
import time

import gym
import pybullet
import pybullet_envs

from mpi4py import MPI
comm = MPI.COMM_WORLD

#from bevodevo.policies.rnns import GatedRNNPolicy
from bevodevo.policies.cnns import ImpalaCNNPolicy
from bevodevo.policies.mlps import MLPPolicy
from bevodevo.algos.es import ESPopulation

class VanillaPolicyGradient():
    """
    Plain Gradient Evolutionary Strategies
    """

    def __init__(self, policy_fn, discrete=False, num_workers=0):
        
        self.policy_fn = policy_fn

        self.lr = 9e-5
        self.verbose = True
        self.num_workers = num_workers
        self.discrete = discrete
        self.std_dev = 1e-1
        self.min_std_dev = 1e-2
        self.decay_std_dev = 0.995
        self.gamma = 0.9
        self.steps_per_epoch = 40000

    def update_policy(self, trajectory):

        batch_size = 1024

        for param in self.my_agent.layers.parameters():
            param.requires_grad = True

        batch_start = 0
        trajectory_length = len(trajectory[0])

        # compute advantages
        advantage = [None] * trajectory_length

        for ii in range(trajectory_length-1, -1, -1):

            if ii == (trajectory_length-1) or trajectory[3][ii]: 
                advantage[ii] = trajectory[2][ii]
            else:
                advantage[ii] = trajectory[2][ii] + self.gamma * advantage[ii+1]

        # normalize advantage
        #advantage = (advantage - np.mean(advantage)) #/ np.std(advantage)
        advantage = np.array(advantage)

        optimizer = torch.optim.Adam(self.my_agent.parameters(), lr=self.lr)

        surrogate_loss = 0.0
        batch_count = 0

        self.my_agent.zero_grad()

        while batch_start < trajectory_length:

            batch_stop = batch_start + max([trajectory_length-batch_start, batch_size])

            observations = torch.Tensor(trajectory[1][batch_start:batch_stop])

            action_mean = self.my_agent.forward(observations)

            action = torch.Tensor(trajectory[0][batch_start:batch_stop])

            advantage_batch = torch.Tensor(advantage[batch_start:batch_stop]).unsqueeze(1)

            # log probability surrogate loss
            surrogate_loss += torch.mean(advantage_batch \
                    * -(torch.abs(action - action_mean)**2) \
                    / (2 * self.std_dev**2))  

            batch_start = batch_stop
            batch_count += 1

        surrogate_loss /= batch_count
        surrogate_loss.backward()
        optimizer.step()

        for param in self.my_agent.layers.parameters():
            param.requires_grad = False

        return surrogate_loss

    def rollout(self, steps=4000):
        
        trajectory = False
        done = True 
        total_steps = 0
        while total_steps < steps:

            if done:
                obs = self.env.reset()
                self.my_agent.reset()

            action = torch.Tensor(self.my_agent.get_action(obs))
            if not self.discrete:
                action += self.std_dev * torch.randn(action.shape) 

            prev_obs = obs
            obs, reward, done, info = self.env.step(action[0])

            if trajectory:
                
                for ii, element in enumerate([action, prev_obs, reward, done]):

                    if type(element) is torch.Tensor:
                        temp = element
                    elif type(element) is np.ndarray:
                        temp = torch.Tensor(element)
                    else:
                        temp = torch.Tensor([element])

                    if len(temp.shape) == 1:
                        temp = temp.unsqueeze(0)

                    trajectory[ii] = torch.cat([trajectory[ii], temp], axis=0)

                trajectory[4].append(info)
            else:
                trajectory = []
                for ii, element in enumerate([action, prev_obs, reward, done]):
                    if type(element) is torch.Tensor:
                        temp = element
                    elif type(element) is np.ndarray:
                        temp = torch.Tensor(element)
                    else:
                        temp = torch.Tensor([element])

                    if len(temp.shape) == 1:
                        temp = temp.unsqueeze(0)
                    trajectory.append(temp)

                trajectory.append([info])

            total_steps += 1

        return trajectory, total_steps

    def mpi_fork(self, n):
        """
        relaunches the current script with workers
        Returns "parent" for original parent, "child" for MPI children
        (from https://github.com/garymcintire/mpi_util/)
        via https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease
        """
        global num_worker, rank
        if n<=1:
            print("if n<=1")
            num_worker = 0
            rank = 0
            return "child"

        if os.getenv("IN_MPI") is None:
            env = os.environ.copy()
            env.update(\
                    MKL_NUM_THREADS="1", \
                    OMP_NUM_THREAdS="1",\
                    IN_MPI="1",\
                    )
            print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
            subprocess.check_call(["mpirun", "-np", str(n), sys.executable] \
            +['-u']+ sys.argv, env=env)

            return "parent"
        else:
            num_worker = comm.Get_size()
            rank = comm.Get_rank()
            return "child"

    def train(self, args):

        my_num_workers = self.num_workers

        if self.mpi_fork(my_num_workers) == "parent":
            os._exit(0)

        if rank == 0:
            self.mantle(args)
        else:
            self.arm(args)


    def mantle(self, args):

        env_name = args.env_name 
        max_epochs = args.generations 
        steps_per_epoch = self.steps_per_epoch
        if num_worker <= 1:
            worker_steps = steps_per_epoch
        else:
            worker_steps = steps_per_epoch // (self.num_workers - 1) 

        disp_every = max(max_epochs // max_epochs, 1)
        save_every = max(max_epochs // max_epochs, 1)

        self.env_fn = gym.make
        self.env_args = env_name 

        hid_dim = [32] 

        # TODO: add seed list loop for experiments here

        self.env = self.env_fn(env_name)
        obs_dim = self.env.observation_space.shape

        if len(obs_dim) == 3:
            obs_dim = obs_dim
        else:
            obs_dim = obs_dim[0]

        try:
            act_dim = self.env.action_space.n
            self.discrete = True
        except:
            act_dim = self.env.action_space.sample().shape[0]
            self.discrete = False
        
        agent_args = {"dim_x": obs_dim, "dim_h": hid_dim, "dim_y": act_dim, "params": None} 

        self.my_agent = self.policy_fn(agent_args)

        self.total_env_interacts = 0

        # prepare performance logging
        exp_id = args.env_name + "_" + args.algorithm[0:6] + str(int(time.time())) 
        res_dir = os.listdir("./results/")
        if args.exp_name not in res_dir:
            os.mkdir("./results/{}".format(args.exp_name))

        results = {"wall_time": []}
        results["total_env_interacts"] = []
        results["generation"] = []
        results["min_fitness"] = []
        results["mean_fitness"] = []
        results["max_fitness"] = []
        results["std_dev_fitness"] = []

        t0 = time.time()
        trajectory = [] 
        print("begin training")
        for epoch in range(max_epochs):

            t1 = time.time()

            # send agents to arm processes
            if num_worker > 0:
                for cc in range(1, num_worker):
                    params = [self.my_agent.get_params(), worker_steps]
                    comm.send(params, dest=cc)

            # apply updates according to _last_ generation's fitness list
            if len(trajectory) > 0 & (epoch % disp_every) == 0:
                loss = self.update_policy(trajectory)
                print("time", time.time()-t0)
                print("epoch {} mean reward {:.2e}+/-{:.2e} max: {:.2e},".\
                      format(epoch, my_mean, my_std_dev, my_max)\
                      + " min: {:.2e}, loss: {:.2e}".format(my_min, loss))

            # for single-core operation, mantle process gathers rollouts
            total_steps = 0
            if num_worker == 0:
                trajectory, steps = self.rollout(steps=worker_steps)
                total_steps += steps
                self.std_dev = max([self.min_std_dev, \
                                    self.std_dev * self.decay_std_dev])

            # receive current generation's fitnesses from arm processes
            if num_worker > 0:
                trajectory = False
                for cc in range(1, num_worker):
                    worker_trajectory = comm.recv(source=cc)

                    if trajectory:
                        for ii in range(4):
                            # add actions, obs, reward, done to trajectory
                            trajectory[ii] = torch.cat([trajectory[ii], worker_trajectory[0][ii]])

                        trajectory[4].extend(worker_trajectory[0][4])

                    else: 
                        trajectory = [None] * 5
                        for ii in range(5):
                            # add actions, obs, reward, done to trajectory
                            trajectory[ii] = worker_trajectory[0][ii]

                    total_steps += worker_trajectory[1]

            self.total_env_interacts += total_steps

            # convert rewards to episodic rewards for reporting
            fitness_list = []
            cumulative_reward = 0.0

            for ii in range(len(trajectory[1])):

                cumulative_reward += trajectory[2][ii]
                if trajectory[3][ii]:
                    fitness_list.append(cumulative_reward)
                    cumulative_reward = 0.0
            
            if cumulative_reward:
                fitness_list.append(cumulative_reward)

            my_min = np.min(fitness_list)
            my_max = np.max(fitness_list)
            my_mean = np.mean(fitness_list) 
            my_std_dev = np.std(fitness_list)

            results["wall_time"].append(time.time() - t0)
            results["total_env_interacts"].append(self.total_env_interacts)
            results["generation"].append(epoch)
            results["min_fitness"].append(my_min)
            results["mean_fitness"].append(my_mean)
            results["max_fitness"].append(my_max)
            results["std_dev_fitness"].append(my_std_dev)

            np.save("results/{}/progress_{}.npy".format(args.exp_name, exp_id), \
                    results, allow_pickle=True)

            if (epoch > 0 and (epoch % save_every == 0)) \
                    or epoch == max_epochs-1:
                
                torch.save(self.my_agent.state_dict(), \
                        "results/{}/best_agent_{}_gen_{}.pt"\
                        .format(args.exp_name, exp_id, epoch))

                elite_params = self.my_agent.get_params()
                np.save("results/{}/best_agent_{}_epoch_{}.npy".\
                        format(args.exp_name, exp_id, epoch),\
                        elite_params)


        for cc in range(1, num_worker):
            print("send shutown signal to worker {}".format(cc))
            comm.send(0, dest=cc)

    def arm(self, args):
 
        env_name = args.env_name 
        
        self.env_fn = gym.make #args.env_fn
        self.env_args = env_name #args.env_name
        self.env = self.env_fn(env_name)

        obs_dim = self.env.observation_space.shape
        if len(obs_dim) == 3:
            obs_dim = obs_dim
        else:
            obs_dim = obs_dim[0]
        hid_dim = 16 #args.hid_dim

        try:
            act_dim = self.env.action_space.n
            self.discrete = True
        except:
            act_dim = self.env.action_space.sample().shape[0]
            self.discrete = False

        while True:
            received = comm.recv(source=0)
            
            if received == 0:
                print("worker {} shutting down".format(rank))
                break
            else:

                params = received[0]
                worker_steps = received[1]

                # may be a problem in some cases when included params in agent_args
                agent_args = {"dim_x": obs_dim, "dim_h": hid_dim, \
                        "dim_y": act_dim, "params": params} 
                self.my_agent = self.policy_fn(agent_args, discrete=self.discrete)
            

            total_substeps = 0

            worker_trajectory, steps = self.rollout(steps=worker_steps)
            total_substeps += steps
            
            comm.send([worker_trajectory, total_substeps], dest=0)
            self.std_dev = max([self.min_std_dev, \
                                self.std_dev * self.decay_std_dev])

if __name__ == "__main__":

    # run tests

    algo = VanillaPolicyGradient
    for policy_fn in [GatedRNNPolicy, MLPPolicy, ImpalaCNNPolicy]:
        temp = algo(policy_fn)

    print("OK")
