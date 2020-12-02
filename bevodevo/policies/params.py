from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import reduce

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import matplotlib.pyplot as plt

class Params():
    """
    policy which outputs the policy parameters directly, i.e. for direct optimization
    """

    def __init__(self, dim_in=7, dim_act=6):
        
        self.dim_act = dim_act

        self.init_params()

    def init_params(self):

        self.params = np.random.randn(self.dim_act)/3 -  1.75
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
