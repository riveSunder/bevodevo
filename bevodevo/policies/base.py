from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import reduce

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import matplotlib.pyplot as plt

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        pass

    def init_params():
        pass

    def reset(self):
        pass

    def forward(self, x):
        pass
      
    def get_params(self):
        pass

    def set_params(self):
        pass


if __name__ == "__main__":

    # run tests
    print("OK")
