import os
import sys

import torch
import numpy as np

from bevodevo.policies.rnns import GatedRNNPolicy
from bevodevo.policies.cnns import ImpalaCNNPolicy
from bevodevo.policies.mlps import MLPPolicy

from bevodevo.algos.es import ESPopulation

class CMAESPopulation(ESPopulation):
    def __init__(self, policy_fn, \
            num_workers=0):

        super(CMAESPopulation, self).__init__(policy_fn)


    def update_pop(self, fitness_list):

        """
        overload ES update function to calculate covariance
        and use multivariate sampling to get params
        """
        sorted_indices = list(np.argsort(fitness_list))
        sorted_indices.reverse()
        sorted_fitness = np.array(fitness_list)[sorted_indices]

        self.elite_pop, _ = self.get_elite(fitness_list, mode=0)

        elite_params = None
        for jj in range(self.elite_keep):

            if elite_params is None:
                elite_params = self.elite_pop[jj].get_params()[np.newaxis,:]
            else:
                elite_params = np.append(elite_params,\
                        self.elite_pop[jj].get_params()[np.newaxis,:],\
                        axis=0)


        temp_params = None
        # TODO: Should calculate the mean for the entire population or 
        # only the population ex elite population?
        
        for kk in range(self.elite_keep, self.population_size):

            if temp_params is None:
                temp_params = self.population[kk].get_params()[np.newaxis,:]
            else:
                temp_params = np.append(temp_params,\
                        self.population[kk].get_params()[np.newaxis,:],\
                        axis=0)

        params_mean = np.mean(elite_params, axis=0)

        covar = (1 / self.elite_keep) \
                * np.matmul((elite_params - self.distribution[0]).T,\
                (elite_params - self.distribution[0]))

        covar = np.clip(covar, -1e1, 1e1)

        var = np.mean( (elite_params - self.distribution[0])**2, axis=0)

        covar_matrix = covar 

        self.distribution = [params_mean, \
                covar_matrix]

        if self.elitism:
            
            for jj in range(self.elite_keep):
                self.population[jj].set_params(self.champions[jj])

            my_start = self.elite_keep
        else:
            my_start = 0

        for ll in range(my_start, self.population_size):
            params = np.random.multivariate_normal(self.distribution[0],\
                    self.distribution[1])

            self.population[ll].set_params(params)

    def get_distribution(self):

        print(self.population[0].num_params)

        return [np.zeros((self.population[0].num_params)),\
                np.eye(self.population[0].num_params)]

if __name__ == "__main__":

    # run tests

    algo = CMAESPopulation
    for policy_fn in [GatedRNNPolicy, MLPPolicy, ImpalaCNNPolicy]:
        temp = algo(policy_fn)

        
    print("OK")
