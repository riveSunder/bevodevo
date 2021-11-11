import os
import sys

import torch
import numpy as np

from bevodevo.policies.rnns import GatedRNNPolicy
from bevodevo.policies.cnns import ImpalaCNNPolicy
from bevodevo.policies.mlps import MLPPolicy

from bevodevo.algos.es import ESPopulation

class NESPopulation(ESPopulation):
    """
    Natural Gradient Evolutionary Strategies
    """

    def __init__(self, policy_fn, discrete=False, num_workers=0):
        super(NESPopulation, self).__init__(policy_fn, discrete = discrete,\
                num_workers = num_workers)
        
        self.lr = 3e-4
        self.std_dev = 1e0
        self.lr_decay = 1. - 1e-4
        self.std_dev_decay = 1. - 1e-7
        self.std_dev_min = 1e-3
        self.lr_min = 1e-6
        self.elite_update = False
        self.tourney_size = 20

    def get_advantage(self, sorted_fitness_list, advantage_mode=3):
        """
            advantage_mode 0 - do nothing, returning sorted_fitness_list as advantage
            advantage_mode 1 - normalize fitness list to have mean 0 and std. dev. of 1.0 
            advantage_mode 2 - shift all fitnesses to be positive and 
                divide by max fitness score 
            advantage_mode 3 - return fitness scores as percentiles 0 to 0.99
            advantage_mode 4 - use a value function as an advantage baseline (not yet
                implemented)
        """

        my_mean = np.mean(sorted_fitness_list)
        my_std_dev = np.std(sorted_fitness_list)
        my_min = np.min(sorted_fitness_list)
        my_max = np.max(sorted_fitness_list - my_min)

        if advantage_mode == 0:
            advantage = sorted_fitness_list
        elif advantage_mode == 1:
            advantage = (sorted_fitness_list - my_mean) / (my_std_dev + 1e-6)
        elif advantage_mode == 2:
            advantage = (sorted_fitness_list - my_min) / my_max 
        elif advantage_mode == 3:
            fit_length = len(sorted_fitness_list)
            advantage = (fit_length - np.arange(fit_length)) / fit_length

        return advantage

    def get_update(self, fitness_list):

        sorted_indices = list(np.argsort(fitness_list))
        sorted_indices.reverse()
        sorted_fitness = np.array(fitness_list)[sorted_indices]

        fitness_mean = np.mean(fitness_list)
        fitness_std = np.std(fitness_list)
        gradient_log_probability = np.zeros_like(self.means)
        gradient_fitness = np.zeros_like(self.means)

        self.elite_pop, elite_fitness = self.get_elite(fitness_list, mode=2)

        if self.elite_update:

            advantage = self.get_advantage(elite_fitness) 

            for mm in range(self.elite_keep):
                
                gradient_log_probability += (1 / self.elite_keep) \
                        * (self.means - self.elite_pop[mm].get_params()) \
                        / self.std_dev**2
                        
                gradient_fitness += advantage[mm] * gradient_log_probability

        else:
            
            advantage = self.get_advantage(sorted_fitness) 

            for mm in range(self.population_size):
                
                gradient_log_probability += (1 / self.population_size) \
                        * (self.means - self.population[mm].get_params()) \
                        / self.std_dev**2
                        
                gradient_fitness += advantage[mm] * gradient_log_probability

        # variable g only used to calculate Fisher matrix
        g = gradient_log_probability[:, np.newaxis]
        fisher_matrix = (1 / self.population_size) * (g.T @ g)


        update = np.linalg.inv(fisher_matrix) * gradient_fitness[:, np.newaxis]

        return update.squeeze() 

        
    def update_pop(self, fitness_list):

        update = self.get_update(fitness_list)
        self.means = self.means - self.lr * update

        if self.elitism:
            
            for jj in range(self.elite_keep):
                self.population[jj].set_params(self.champions[jj]) 

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

    algo = NESPopulation
    for policy_fn in [GatedRNNPolicy, MLPPolicy, ImpalaCNNPolicy]:
        temp = algo(policy_fn)

        
    print("OK")
