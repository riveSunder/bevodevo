import os
import sys
import subprocess

import torch
import numpy as np
import time

from bevodevo.policies.rnns import GatedRNNPolicy
from bevodevo.policies.cnns import ImpalaCNNPolicy
from bevodevo.policies.mlps import MLPPolicy

from bevodevo.algos.es import ESPopulation

class GeneticPopulation(ESPopulation):

    def __init__(self, policy_fn, num_workers=0):
        super(GeneticPopulation, self).__init__(policy_fn, num_workers=num_workers)


    def mutate(self, params):

        # mutation parameters
        pruning_chance = 1.0#  0.33
        weight_chance = 0.0 #0.33
        recombination_chance = 0.0 #0.33

        pruning_rate = 0.025
        weight_var = 1e-1
        recombination_rate = 0.125

        mut_chance = np.random.random((3))
        
        mut_params = 1.0 * params
        if mut_chance[0] <= pruning_chance:
            # deletions (pruning)
            mut_params = params * (np.random.random((params.shape)) < pruning_rate)

        if mut_chance[1] <= weight_chance:
            # weight value mutations 
            mut_params = mut_params + np.random.randn(params.shape[0],) * weight_var

        if mut_chance[2] <= recombination_chance:
            # recombination 
            a = np.arange(self.elite_keep)
            p = np.exp(1/(a+1)) / np.sum(np.exp(1/(a+1)))

            idx = np.random.choice(a, p = p)
            reco_params = self.elite_pop[idx].get_params()

            recombinations = np.random.random((params.shape))
            mut_params[recombinations < recombination_rate] = \
                    reco_params[recombinations < recombination_rate]

        return mut_params


    def update_pop(self, fitness_list):

        sorted_indices = list(np.argsort(fitness_list))
        sorted_indices.reverse()
        sorted_fitness = np.array(fitness_list)[sorted_indices]

        self.elite_pop, elite_fitness = self.get_elite(fitness_list, mode=1)


        if self.elitism:
            
            for jj in range(self.elite_keep):
                self.population[jj] = self.champions[jj]

            my_start = self.elite_keep
        else:
            my_start = 0

        # sample according to the fitness ranking, softmax ensures sum(p) = 1.0
        softmax = lambda x: np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)), axis=0)

        # p for sampling probability
        p = softmax(np.array(elite_fitness))
        # a for indices to sample accord to fit. prob. p
        a = np.arange(self.elite_keep)

        for kk in range(my_start, self.population_size):

            idx = np.random.choice(a, p=p)
            params = self.elite_pop[idx].get_params()

            mut_params = self.mutate(params)
            self.population[kk].set_params(mut_params)

if __name__ == "__main__":

    # run tests

    algo = GeneticPopulation
    for policy_fn in [GatedRNNPolicy, MLPPolicy, ImpalaCNNPolicy]:
        temp = algo(policy_fn)

        
    print("OK")
