import os
import sys
import subprocess

import torch
import numpy as np

from bevodevo.policies.rnns import GatedRNNPolicy
from bevodevo.policies.cnns import ImpalaCNNPolicy
from bevodevo.policies.mlps import MLPPolicy

from bevodevo.algos.es import ESPopulation

class RandomSearch(ESPopulation):

    def __init__(self, policy_fn, num_workers=0):
        super(RandomSearch, self).__init__(policy_fn, num_workers=num_workers)

        self.std_dev = 1e0
        self.elitism = True

    def update_pop(self, fitness_list):

        sorted_indices = list(np.argsort(fitness_list))
        sorted_indices.reverse()
        sorted_fitness = np.array(fitness_list)[sorted_indices]

        self.elite_pop, elite_fitness = self.get_elite(fitness_list)

        for nn in range(self.elite_keep):
            self.population[nn].set_params(self.champions[nn])

        for kk in range(self.elite_keep, self.population_size):
            agent_params = np.random.randn(self.population[kk].num_params) \
                    * self.std_dev 

            self.population[kk].set_params(agent_params)

if __name__ == "__main__":

    # run tests

    algo = RandomSearch
    for policy_fn in [GatedRNNPolicy, MLPPolicy, ImpalaCNNPolicy]:
        temp = algo(policy_fn)

        
    print("OK")
