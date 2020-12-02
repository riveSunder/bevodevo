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

from bevodevo.policies.rnns import GatedRNNPolicy
from bevodevo.policies.cnns import ImpalaCNNPolicy
from bevodevo.policies.mlps import MLPPolicy

from bevodevo.algos.es import ESPopulation

class PGESPopulation(ESPopulation):
    """
    Plain Gradient Evolutionary Strategies
    """

    def __init__(self, policy_fn, discrete=False, num_workers=0):
        super(PGESPopulation, self).__init__(policy_fn, discrete = discrete,\
                num_workers = num_workers)
        
        self.lr = 1e-5
        self.std_dev = 1e0
        self.lr_decay = 1. - 1e-4
        self.std_dev_decay = 1. - 1e-7
        self.std_dev_min = 1e-3
        self.lr_min = 1e-6
        self.elite_update = True

    def get_update(self, fitness_list):

        sorted_indices = list(np.argsort(fitness_list))
        sorted_indices.reverse()
        sorted_fitness = np.array(fitness_list)[sorted_indices]

        fitness_mean = np.mean(fitness_list)
        fitness_std = np.std(fitness_list)

        self.elite_pop, elite_fitness = self.get_elite(fitness_list, mode=2)

        if self.elite_update: 
            
            advantage = (elite_fitness - fitness_mean) / (fitness_std + 1e-6)

            update = np.zeros_like(self.means)
                
            for mm in range(self.elite_keep):
                
                update += (self.means - advantage[mm] * self.elite_pop[mm].get_params())\
                        / self.elite_pop[mm].var**2


            update = update / (self.population_size)

        else:

            advantage = (fitness_list - fitness_mean) / (fitness_std + 1e-6)

            update = np.zeros_like(self.means)
                
            for mm in range(self.population_size):
                
                update += (self.means - advantage[mm] * self.population[mm].get_params())\
                        / self.population[mm].var**2

            update = update / (self.population_size)

        return  update

        
    def update_pop(self, fitness_list):

        update = self.get_update(fitness_list)
        self.means = self.means + self.lr * update

        if self.elitism:
            
            for jj in range(self.elite_keep):
                self.population[jj] = self.champions[jj]

            my_start = self.elite_keep
        else:
            my_start = 0

        for kk in range(my_start, self.population_size):
            agent_params = self.means + self.std_dev \
                    * np.random.randn(self.means.shape[0]) 

            self.population[kk].set_params(agent_params)

        self.std_dev = max([self.std_dev_min, self.std_dev * self.std_dev_decay])
        self.lr = max([self.lr_min, self.lr * self.lr_decay])

if __name__ == "__main__":

    # run tests

    algo = PGESPopulation
    for policy_fn in [GatedRNNPolicy, MLPPolicy, ImpalaCNNPolicy]:
        temp = algo(policy_fn)

    print("OK")
