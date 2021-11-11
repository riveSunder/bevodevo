from functools import reduce

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Params():
    """
    policy which outputs the policy parameters directly, i.e. for direct optimization
    """

    def __init__(self, **kwargs): 
        
        self.dim_act = kwargs["dim_act"] if "dim_act" in kwargs.keys() else 6
        
        self.means = np.zeros((self.dim_act))
        self.standard_deviation = np.array([0.33] * self.dim_act)

        self.init_params()

    def init_params(self):

        self.params = self.standard_deviation * np.random.randn(self.dim_act) \
                + self.means

        self.num_params = self.dim_act

    def forward(self, obs):
        return self.get_params()

    def get_params(self):
        return self.params

    def set_params(self, params):
        assert params.shape == self.params.shape

        self.params = params 

    def reset(self):
        pass

if __name__ == "__main__":

    # run tests
    print("OK")
