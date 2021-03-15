from abc import ABC, abstractmethod
import os
import sys
import subprocess
import copy

import torch
import numpy as np
import time

import gym
import pybullet
import pybullet_envs


from mpi4py import MPI
comm = MPI.COMM_WORLD

from bevodevo.policies.rnns import GatedRNNPolicy
from bevodevo.policies.cnns import ImpalaCNNPolicy
from bevodevo.policies.mlps import MLPPolicy

class ESPopulation:

    def __init__(self, policy_fn, discrete=False, num_workers=0, threshold=float("Inf")):
        
        self.policy_fn = policy_fn

        self.verbose = True
        self.num_workers = num_workers
        self.discrete = discrete

        self.elitism = True
        self.champions = None
        self.leaderboard = None

        self.abort = False
        self.threshold = threshold

        self.tourney_size = 5

    def get_agent_action(self, obs, agent_idx):
        return self.population[agent_idx].get_action(obs)

    def get_fitness(self, agent_idx, epds=4, render=False, view_elite=False):
        fitness = []
        sum_rewards = []
        total_steps = 0

        if view_elite:
            agent_idx = 0
            #self.env.render(mode="human")
            self.env = self.env_fn(self.env_args, render_mode="human")

        self.population[agent_idx].reset()

        for epd in range(epds):

            obs = self.env.reset()
            prev_obs = None
            done = False
            sum_reward = 0.0
            while not done and not(self.abort):
                action = self.get_agent_action(obs, agent_idx)

                if type(action) == np.ndarray: #len(action.shape) > 1:
                    action = action[0]

                if not(self.discrete):

                    if np.isnan(action).any():
                        print("warning, nan encountered in action")
                        action = np.nan_to_num(action, 0.0)
                        self.abort = True

                    if (np.max(action) > self.env.action_space.high).any()\
                            or (np.min(action) < self.env.action_space.low).any():
                        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

                prev_obs = obs
                try:
                    obs, reward, done, info = self.env.step(action)
                except:
                    print("something went wrong?")
                    import pdb; pdb.set_trace()

                if len(obs.shape) == 3:
                    obs = obs / 255.
                    if prev_obs is not None:
                        obs = 1.5 * obs - 0.5 * prev_obs
                else:
                    obs = torch.Tensor(obs).unsqueeze(0)

                sum_reward += reward
                total_steps += 1

            sum_rewards.append(sum_reward)

        fitness = np.sum(sum_rewards) / epds

        if(0):
            if view_elite:
                # this part doesn't work. (apparently both calls are deprecated)
                self.env.close()
                self.env.render(mode="close")

        return fitness, total_steps

    def get_elite(self, fitness_list, mode=0):
        """
        select elite population according to fitness using one of three methods:
            0 - truncation. Select the top self.elite_keep policies   
            1 - fitness proportional. Randomly select policies with 
                probability proportional to their fitness
            2 - tournament selection. Select policies that perform best against
                a limited number of their neighbors 
        """

        if mode == 0:
            # truncation selection

            sorted_indices = list(np.argsort(fitness_list))
            sorted_indices.reverse()
            sorted_fitness = np.array(fitness_list)[sorted_indices]

            elite_pop = []
            elite_fitness = []

            for jj in range(self.elite_keep):

                elite_pop.append(self.population[sorted_indices[jj]])
                elite_fitness.append(fitness_list[sorted_indices[jj]])

        elif mode == 1:
            # fitness proportional selection

            # apply softmax to fitness scores to get sampling probabilities
            softmax = lambda x: np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)), axis=0)

            # p for sampling probability
            p = softmax(np.array(fitness_list))
            # a for indices to sample accord to fit. prob. p
            a = np.arange(self.population_size)

            sorted_indices = list(np.random.choice(a, p=p, size=self.elite_keep))
            sorted_fitness = np.array(fitness_list)[sorted_indices]

            elite_pop = []
            elite_fitness = []

            for jj in range(self.elite_keep):

                elite_pop.append(self.population[sorted_indices[jj]])
                elite_fitness.append(fitness_list[sorted_indices[jj]])

        elif mode == 2:
            # tournament selection 

            elite_pop = []
            elite_fitness = []
            
            tourney_start = 0
            iteration = 0

            while len(elite_pop) < self.elite_keep:
                if tourney_start >= self.population_size:
                    iteration += 1
                    tourney_start = iteration % self.tourney_size


                sorted_indices = list(\
                        np.argsort(fitness_list[tourney_start:tourney_start+self.tourney_size]))

                elite_pop.append(self.population[sorted_indices[-1]])
                elite_fitness.append(fitness_list[sorted_indices[-1]])

                tourney_start += self.tourney_size

        # populate an all-generations champions leaderboard if elitism is true

        if self.elitism:
            if self.champions is None:
                self.champions = copy.deepcopy(elite_pop)
                self.leaderboard = copy.deepcopy(elite_fitness)
            else:
                sorted_indices = list(np.argsort(fitness_list))
                sorted_indices.reverse()
                sorted_fitness = np.array(fitness_list)[sorted_indices]

                for oo in range(self.elite_keep):
                    for pp in range(self.elite_keep):

                        if sorted_fitness[oo] >= self.leaderboard[pp]:
                            self.leaderboard.insert(pp, sorted_fitness[oo])
                            self.champions.insert(pp, self.population[sorted_indices[oo]])
                            sorted_fitness[oo] = -float("Inf")

            self.leaderboard = self.leaderboard[:self.elite_keep]
            self.champions = self.champions[:self.elite_keep]

        return elite_pop, elite_fitness 

    def update_pop(self, fitness_list):

        self.elite_pop, _ = self.get_elite(fitness_list)

        params = None
        for jj in range(self.elite_keep):

            if params is None:
                params = self.elite_pop[jj].get_params()[np.newaxis,:]
            else:
                params = np.append(params,\
                        self.elite_pop[jj].get_params()[np.newaxis,:],\
                        axis=0)

        params_mean = np.mean(params, axis=0)
        self.means = params_mean

        if self.elitism:
            
            for jj in range(self.elite_keep):
                self.population[jj] = self.champions[jj]

            my_start = self.elite_keep
        else:
            my_start = 0

        for kk in range(my_start, self.population_size):
            agent_params = params_mean + \
                    np.random.randn(params_mean.shape[0]) \
                    * np.sqrt(self.population[kk].var)

            self.population[kk].set_params(agent_params)

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

    def get_distribution(self):

        return [np.zeros((self.population[0].num_params)),\
                np.ones(self.population[0].num_params)]

    def mantle(self, args):

        env_name = args.env_name 
        max_generations = args.generations 
        population_size = args.population_size 

        disp_every = max(max_generations // 100, 1)
        save_every = max(max_generations // 20, 1)

        self.env_fn = gym.make #args.self.env_fn
        self.env_args = env_name #args.env_name

        hid_dim = 16 #[32, 32] #args.hid_dims

        seeds = args.seeds
        self.threshold = args.performance_threshold

        self.env = self.env_fn(self.env_args) 
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

        self.population_size = population_size
        self.elite_keep = int(0.125 * self.population_size)
        
        agent_args = {"dim_x": obs_dim, "dim_h": hid_dim, "dim_y": act_dim, "params": None} 



        for seed in seeds:

            self.champions = None
            self.leaderboard = None
            self.abort = False
            # seed everything
            my_seed = seed
            np.random.seed(my_seed)
            torch.random.manual_seed(my_seed)

            self.population = [self.policy_fn(agent_args, discrete=self.discrete)\
                    for ii in range(self.population_size)]

            self.total_env_interacts = 0

            self.means = self.get_distribution()[0]

            self.distribution = self.get_distribution()

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
            results["args"] = str(args)

            fitness_list = []
            t0 = time.time()
            generation = 0
            threshold_count = 0
            print("begin evolution with seed {}".format(seed))
            while (generation <= max_generations) and self.abort == False:

                t1 = time.time()

                # apply updates according to _last_ generation's fitness list
                if len(fitness_list) > 0:
                    print("time", time.time()-t0)
                    print("gen {} mean fitness {:.2e}+/-{:.2e} max: {:.2e}, min: {:.2e}"\
                            .format(generation, my_mean, my_std_dev, \
                            my_max, my_min))
                    self.update_pop(fitness_list)

                # send agents to arm processes
                if num_worker > 0:

                    subpop_size = int(self.population_size / (num_worker-1))
                    pop_remainder = self.population_size % (num_worker-1)
                    pop_left = self.population_size


                    batch_end = 0
                    extras = 0
                    for cc in range(1, num_worker):
                        batch_size = min(subpop_size, pop_left)

                        if pop_remainder:
                            batch_size += 1
                            pop_remainder -= 1
                            extras += 1

                        batch_start = batch_end 
                        batch_end = batch_start + batch_size 

                        params_list = [my_agent.get_params() \
                                for my_agent in self.population[batch_start:batch_end]]

                        #print("send params of len {} to worker {}".format(len(params_list), cc))
                        comm.send(params_list, dest=cc)


                # for single-core operation, mantle process gathers rollouts
                total_steps = 0
                if num_worker == 0:
                    fitness_list = []
                    for agent_idx in range(self.population_size):
                        fitness, steps = self.get_fitness(agent_idx)

                        fitness_list.append(fitness)
                        total_steps += steps

                # receive current generation's fitnesses from arm processes
                if num_worker > 0:
                    fitness_list = []
                    pop_left = self.population_size
                    for cc in range(1, num_worker):
                        fit = comm.recv(source=cc)
                        fitness_list.extend(fit[0])
                        
                        #print("worker {} returns fitness list of len {}".format(cc, len(fit[0])))
                        total_steps += fit[1]

                self.total_env_interacts += total_steps

                my_min = np.min(fitness_list)
                my_max = np.max(fitness_list)
                my_mean = np.mean(fitness_list)
                my_std_dev = np.std(fitness_list)

                results["wall_time"].append(time.time() - t0)
                results["total_env_interacts"].append(self.total_env_interacts)
                results["generation"].append(generation)
                results["min_fitness"].append(my_min)
                results["mean_fitness"].append(my_mean)
                results["max_fitness"].append(my_max)
                results["std_dev_fitness"].append(my_std_dev)

                np.save("results/{}/progress_{}_s{}.npy".format(args.exp_name, exp_id, seed),\
                        results, allow_pickle=True)

                if my_max >= self.threshold:
                    threshold_count += 1
                else:
                    threshold_count = 0

                if threshold_count >= 5:
                    print("performance threshold met, ending training")
                    print(self.means)
                    self.abort = True

                if (generation > 0 and (generation % save_every == 0)) \
                        or generation == max_generations-1\
                        or self.abort:
                    

                    torch.save(self.elite_pop[0].state_dict(), \
                            "results/{}/best_agent_{}_gen_{}_s{}.pt"\
                            .format(args.exp_name, exp_id, generation, seed))

                    if self.elitism:

                        elite_params = {}
                        for ii, elite in enumerate(self.champions):
                            elite_params["elite_{}".format(ii)] = elite.get_params()
                    else:
                        elite_params = {}
                        for ii, elite in enumerate(self.elite_pop):
                            elite_params["elite_{}".format(ii)] = elite.get_params()



                    elite_params["env_name"] = env_name
                    np.save("results/{}/elite_pop_{}_gen_{}_s{}".\
                            format(args.exp_name, exp_id, generation, seed),\
                            elite_params)

                generation += 1
                


        for cc in range(1, num_worker):
            print("send shutown signal to worker {}".format(cc))
            comm.send(0, dest=cc)

    def arm(self, args):
 
        env_name = args.env_name 
        
        self.env_fn = gym.make #args.env_fn
        self.env_args = env_name #args.env_name
        self.env = self.env_fn(self.env_args)

        obs_dim = self.env.observation_space.shape
        if len(obs_dim) == 3:
            obs_dim = obs_dim
        else:
            obs_dim = obs_dim[0]
        hid_dim = 16 #[32,32] #args.hid_dim

        try:
            act_dim = self.env.action_space.n
            self.discrete = True
        except:
            act_dim = self.env.action_space.sample().shape[0]
            self.discrete = False

        while True:
            params_list = comm.recv(source=0)
            
            if params_list == 0:
                print("worker {} shutting down".format(rank))
                break

            self.population_size = len(params_list)

            self.population = [] 

            for ii in range(self.population_size):

                agent_args = {"dim_x": obs_dim, "dim_h": hid_dim, \
                        "dim_y": act_dim, "params": None} 
                self.population.append(self.policy_fn(agent_args, discrete=self.discrete))
                self.population[-1].set_params(params_list[ii])
            

            fitness_sublist = []
            total_substeps = 0
            for agent_idx in range(self.population_size):
                fitness, steps = self.get_fitness(agent_idx)
                
                fitness_sublist.append(fitness)
                total_substeps += steps

            comm.send([fitness_sublist, total_substeps], dest=0)

class ConstrainedESPopulation(ESPopulation):

    def __init__(self, \
            policy_fn, \
            discrete=False, \
            num_workers=0, \
            threshold=float("Inf")):
        super(ConstrainedESPopulation, self).__init__(policy_fn, \
                discrete=discrete,\
                num_workers=num_workers,\
                threshold=threshold)

    #overload the following inheriting functions
    def get_fitness(self, agent_idx, epds=4, render=False, view_elite=False):
        sum_rewards = []
        sum_costs = []
        total_steps = 0

        if view_elite:
            agent_idx = 0
            #self.env.render(mode="human")
            self.env = self.env_fn(self.env_args, render_mode="human")

        self.population[agent_idx].reset()

        for epd in range(epds):

            obs = self.env.reset()
            prev_obs = None
            done = False
            sum_reward = 0.0
            sum_cost = 0.0
            while not done and not(self.abort):
                action = self.get_agent_action(obs, agent_idx)

                if type(action) == np.ndarray: #len(action.shape) > 1:
                    action = action[0]

                if not(self.discrete):

                    if np.isnan(action).any():
                        print("warning, nan encountered in action")
                        action = np.nan_to_num(action, 0.0)
                        self.abort = True

                    if (np.max(action) > self.env.action_space.high).any()\
                            or (np.min(action) < self.env.action_space.low).any():
                        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

                prev_obs = obs
                try:
                    obs, reward, done, info = self.env.step(action)
                except:
                    print("something went wrong?")
                    import pdb; pdb.set_trace()

                if len(obs.shape) == 3:
                    obs = obs / 255.
                    if prev_obs is not None:
                        obs = 1.5 * obs - 0.5 * prev_obs
                else:
                    obs = torch.Tensor(obs).unsqueeze(0)

                sum_reward += reward
                sum_cost += info["cost"]
                total_steps += 1

            sum_rewards.append(sum_reward)
            sum_costs.append(sum_cost)

        fitness = np.sum(sum_rewards) / epds
        cost = np.sum(sum_costs) / epds

        if(0):
            if view_elite:
                # this part doesn't work. (apparently both calls are deprecated)
                self.env.close()
                self.env.render(mode="close")

        return fitness, total_steps, cost

    def update_pop(self, fitness_list):
        pass

        #needs to sort by cost first before reward based fitness

    def mantle(self, args):
        pass

        # needs to collect cost with fitness from arms and pass to `update_pop`

    def arm(self, args):
        pass

        # needs to return fitness and cost lists

if __name__ == "__main__":

    # run tests

    algo = ConstrainedESPopulation
    for policy_fn in [GatedRNNPolicy, MLPPolicy, ImpalaCNNPolicy]:
        temp = algo(policy_fn)

        
    print("OK")
