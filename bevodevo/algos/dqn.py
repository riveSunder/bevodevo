from abc import ABC, abstractmethod
import os
import sys
import subprocess

import torch
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

class DQN(ESPopulation):

    def __init__(self, policy_fn, agent_args=None, num_workers=0):
        
        super(DQN, self).__init__(policy_fn)

        # use mpi_fork and train from ESPopulation 

        if agent_args is None:
            self.agent_args = agent_args = {\
#                    "dim_x": (64,64,3),\
                    "dim_x": (210,160,3),\
                    "dim_h": [32], \
                    "dim_y": 15, \
                    "params": None\
                    } 

        self.policy_fn = policy_fn

        self.verbose = True
        self.num_workers = num_workers


        # hyperparameters
        self.min_eps = 0.001 #torch.Tensor(np.array(0.02))
        self.eps = 0.95 #torch.Tensor(np.array(0.9))
        self.eps_decay = 0.99
        self.lr = 1e-4
        self.batch_size = 32
        self.steps_per_epoch = 20000 #8192 #16384 #4096 #24096
        self.updates_per_epoch = self.steps_per_epoch // self.batch_size

        self.steps_per_worker = self.steps_per_epoch // (num_workers+1)
        self.update_qt = 10
        self.epochs = 30000
        self.device = torch.device("cpu")
        #torch.device("cuda:0") \
        #        if torch.cuda.is_available() else torch.device("cpu")
        self.discount = 0.99
        self.difficulty = "easy"
        

    def get_episodes(self, steps=None, render=False):

        if steps is None:
            steps = self.steps_per_worker

        # initialize trajectory buffer
        l_obs = torch.Tensor()
        l_rew = torch.Tensor()
        l_act = torch.Tensor()
        l_next_obs = torch.Tensor()
        l_done = torch.Tensor()

        done = True
        prev_obs = None
        with torch.no_grad():
            for step in range(steps):

                if step > 100: render = False
                if done:
                    prev_obs = None
                    obs = self.env.reset()
                    done = False

                    if len(obs.shape) == 3:
                        obs = torch.Tensor(obs).permute(2,0,1).unsqueeze(0)
                        obs = obs / obs.max()
                    else:
                        obs = torch.Tensor(obs).unsqueeze(0)

                if torch.rand(1) < self.eps:
                    action = self.env.action_space.sample()
                else:
                    q_values = self.q(obs)
                    # better to sample stochastically from q_values?
                    act = torch.argmax(q_values, dim=-1)
                    action = act.detach().cpu().numpy()[0]
                

                if render:
                    time.sleep(0.0001)
                    self.env.render()

                prev_obs = obs

                obs, reward, done, info = self.env.step(action)

                if len(obs.shape) == 3:
                    obs = torch.Tensor(obs).permute(2,0,1).unsqueeze(0)
                    obs = obs / obs.max()
                    if prev_obs is not None:
                        obs = 1.5 * obs - 0.5 * prev_obs
                else:
                    obs = torch.Tensor(obs).unsqueeze(0)

                # for recurrent networks, obs should include cell state/hidden state 
                l_obs = torch.cat([l_obs, prev_obs], dim = 0)
                l_next_obs = torch.cat([l_next_obs, obs], dim=0)

                l_rew = torch.cat([l_rew, \
                        torch.Tensor(np.array(1. * reward)).reshape(1,1)], dim=0)
                l_act = torch.cat([l_act, \
                        torch.Tensor(np.array(1. * action)).reshape(1,1)], dim=0)
                l_done = torch.cat([l_done, \
                        torch.Tensor(np.array(1. * done)).reshape(1,1)], dim=0)

        return l_obs, l_act, l_rew, l_next_obs, l_done            


    def compute_q_loss(self, l_obs, l_act, l_rew, l_next_obs, l_done, debug=False):

        #loss_fn = torch.nn.MSELoss()
        loss_fn = lambda y_pred, y_tgt: torch.mean(\
                0.8*(y_tgt-y_pred)**2 + 0.2*torch.abs(y_tgt-y_pred))

        if (0):
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.q.to(device)
                self.qt.to(device)
                l_obs = l_obs.to(device)
                l_act = l_act.to(device)
                l_rew = l_rew.to(device)
                l_next_obs = l_next_obs.to(device)
                l_done = l_done.to(device)

        if (debug==True):
            import pdb; pdb.set_trace()

        with torch.no_grad():

            qt_values = self.qt.forward(l_next_obs)

            qt_next = self.q.forward(l_next_obs)
            qt_max = torch.gather(qt_values, -1,\
                    torch.argmax(qt_next, dim=-1).unsqueeze(-1))

            value_tgt = l_rew + ((1. - l_done) * self.discount * qt_max)

        l_act = l_act.long()
        q_values = self.q(l_obs)

        q_act_value = torch.gather(q_values, -1, l_act)

        loss = loss_fn(q_act_value, value_tgt)

        return loss

    def mantle(self, args):

        max_generations = args.generations 
        disp_every = max(max_generations // 100, 1)

        self.env_fn = gym.make #args.self.env_fn

        self.env = self.env_fn(args.env_name)# num_levels=100,\
        #        start_level=1337, use_sequential_levels=True)
        
        obs_dim = self.env.observation_space.shape
        if len(obs_dim) == 3:
            obs_dim = obs_dim
        else:
            obs_dim = obs_dim[0]

        try:
            act_dim = self.env.action_space.n
            discrete = True
        except:
            act_dim = self.env.action_space.sample().shape[0]
            discrete = False

        self.agent_args["dim_x"] = obs_dim
        self.agent_args["dim_y"] = act_dim

        self.q = self.policy_fn(self.agent_args, discrete=True, use_grad=True)
        self.qt = self.policy_fn(self.agent_args, discrete=True, use_grad=True)
        self.q.to(torch.device("cpu")) #self.device)
        self.qt.to(torch.device("cpu")) #self.device)

        for param in self.qt.parameters():
            param.requires_grad = False


        self.total_env_interacts = 0
        self.total_time = 0.

        optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)

        t0 = time.time()
        print("steps_per_worker", self.steps_per_worker)
        for generation in range(max_generations):

            t1 = time.time()

            # send out q and qt parameters
            self.q.to(torch.device("cpu")) #self.device)
            self.qt.to(torch.device("cpu")) #self.device)
            if self.num_workers > 0:

                for cc in range(1, self.num_workers):

                    policy_list = [self.q.state_dict(), self.qt.state_dict()]

                    comm.send(policy_list, dest=cc)
                    
            # receive q parameters
            if self.num_workers > 0:
                my_debug=False

                l_obs = torch.Tensor()
                l_act = torch.Tensor()
                l_rew = torch.Tensor()
                l_next_obs = torch.Tensor()
                l_done = torch.Tensor()

                for cc in range(1, self.num_workers):
                    trajectory = comm.recv(source=cc)
                    l_obs = torch.cat([l_obs, trajectory[0]], dim=0)
                    l_act = torch.cat([l_act, trajectory[1]], dim=0)
                    l_rew = torch.cat([l_rew, trajectory[2]], dim=0) 
                    l_next_obs = torch.cat([l_next_obs, trajectory[3]], dim=0)
                    l_done = torch.cat([l_done, trajectory[4]], dim=0) 
            else:
                if generation % 150 == 0: 
                    l_obs, l_act, l_rew, l_next_obs, l_done = self.get_episodes(render=False)
                    my_debug=False
                else:
                    l_obs, l_act, l_rew, l_next_obs, l_done = self.get_episodes()
                    my_debug=False



            self.eps = max(self.min_eps, self.eps * self.eps_decay)

            # train q 
            if 0: #torch.cuda.is_available():
                device = torch.device("cuda")
                self.q.to(device)
                self.qt.to(device)
                l_obs = l_obs.to(device)
                l_act = l_act.to(device)
                l_rew = l_rew.to(device)
                l_next_obs = l_next_obs.to(device)
                l_done = l_done.to(device)

            loss_mean = 0.0
            
            for update in range(self.updates_per_epoch):
                idx = torch.randint(l_obs.shape[0], (self.batch_size,))

                self.q.zero_grad()

                loss = self.compute_q_loss(\
                        l_obs[idx],\
                        l_act[idx],\
                        l_rew[idx],\
                        l_next_obs[idx],\
                        l_done[idx], debug=my_debug)

                loss.backward()
                optimizer.step()

                loss_mean += loss.detach().cpu().numpy()

            loss_mean /= self.updates_per_epoch

            t2 = time.time()
            print("{:.2f} s epoch {}, mean loss: {:.3e}, mean rew: {:.2e}".format(\
                    t2-t0, generation, loss_mean, torch.sum(l_rew)/torch.sum(l_done)), self.eps)


            # update qt periodically
            if generation % self.update_qt == 0:
                #self.qt.load_state_dict(copy.deepcopy(self.q.state_dict()))
                self.qt.load_state_dict(self.q.state_dict())



        if self.num_workers > 0:
            for cc in range(1, self.num_workers):
                print("send shutown signal to worker {}".format(cc))
                comm.send(0, dest=cc)

    def arm(self, args):
        self.env_fn = gym.make #args.self.env_fn

        self.env = self.env_fn(args.env_name) #, num_levels=100,\
        #        start_level=1337, use_sequential_levels=True)
        obs_dim = self.env.observation_space.shape

        if len(obs_dim) == 3:
            obs_dim = obs_dim
        else:
            obs_dim = obs_dim[0]

        try:
            act_dim = self.env.action_space.n
            discrete = True
        except:
            act_dim = self.env.action_space.sample().shape[0]
            discrete = False

        self.agent_args["dim_x"] = obs_dim
        self.agent_args["dim_y"] = act_dim

#        self.q = self.policy_fn(self.agent_args)
#        self.qt = self.policy_fn(self.agent_args)
        self.q = self.policy_fn(self.agent_args, discrete=True)
        self.qt = self.policy_fn(self.agent_args, discrete=True)
        self.q.to(self.device)
        self.qt.to(self.device)

        for param in self.qt.parameters():
            param.requires_grad = False
        self.total_env_interacts = 0
        self.total_time = 0.

        t0 =  time.time()
        while True:
            param_list = comm.recv(source=0)

            if param_list == 0:
                print("worker shutting down")
                break

            self.q.load_state_dict(param_list[0])
            self.qt.load_state_dict(param_list[1])

            # get episodes
            l_obs, l_act, l_rew, l_next_obs, l_done = self.get_episodes()
            self.eps = max(self.min_eps, self.eps * self.eps_decay)

            # send q
            trajectory = [\
                    l_obs,\
                    l_act,\
                    l_rew,\
                    l_next_obs,\
                    l_done\
                    ]

            comm.send(trajectory, dest=0)




if __name__ == "__main__":

    # run tests

    algo = DQN 
    for policy_fn in [GatedRNNPolicy, MLPPolicy, ImpalaCNNPolicy]:
        temp = algo(policy_fn)

        
    print("OK")
