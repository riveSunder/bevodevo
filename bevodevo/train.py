import os
import sys
import argparse
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
from bevodevo.policies.mlps import MLPPolicy, CPPNMLPPolicy, CPPNHebbianMLP,\
        HebbianMLP, ABCHebbianMLP

from bevodevo.algos.es import ESPopulation
from bevodevo.algos.cmaes import CMAESPopulation
from bevodevo.algos.pges import PGESPopulation
from bevodevo.algos.nes import NESPopulation
from bevodevo.algos.ga import GeneticPopulation
from bevodevo.algos.random_search import RandomSearch

#from bevodevo.algos.vpg import VanillaPolicyGradient
#from bevodevo.algos.dqn import DQN


def train(argv):

    # env_name, generations, population_size, 
    
    if "gatedrnn" in argv.policy.lower():
        policy_fn = GatedRNNPolicy
        argv.policy = "GatedRNNPolicy" 
    elif "impala" in argv.policy.lower():
        policy_fn = ImpalaCNNPolicy
        argv.policy = "ImpalaCNNPolicy"
    elif "cppnmlp" in argv.policy.lower():
        policy_fn = CPPNMLPPolicy
        arg.policy = "CPPNMLPPolicy"
    elif "abchebbianmlp" in argv.policy.lower():
        policy_fn = ABCHebbianMLP
        argv.policy = "ABCHebbianMLP"
    elif "cppnhebbianmlp" in argv.policy.lower():
        policy_fn = CPPNHebbianMLP
        argv.policy = "CPPNHebbianMLP"
    elif "hebbianmlp" in argv.policy.lower():
        policy_fn = HebbianMLP
        argv.policy = "HebbianMLP"
    elif "mlppolicy" in argv.policy.lower():
        policy_fn = MLPPolicy
        argv.policy = "MLPPolicy"
    else:
        assert False, "policy not found, check spelling?"

    if "ESPopulation" == argv.algorithm:
        population_fn = ESPopulation
    elif "CMAESPopulation" == argv.algorithm:
        population_fn = CMAESPopulation
    elif "Genetic" in argv.algorithm:
        population_fn = GeneticPopulation
    elif "PGES" in argv.algorithm:
        population_fn = PGESPopulation
    elif "NES" in argv.algorithm:
        population_fn = NESPopulation
    elif "dqn" in argv.algorithm:
        population_fn = DQN
    elif "vpg" in argv.algorithm.lower():
        population_fn = VanillaPolicyGradient
    elif "andom" in argv.algorithm:
        population_fn = RandomSearch
    else:
        assert False, "population algo not found, check spelling?"

    num_workers = argv.num_workers

    population = population_fn(policy_fn, num_workers=num_workers)
    
    population.train(argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Experiment parameters")
    parser.add_argument("-n", "--env_name", type=str, \
            help="name of environemt", default="InvertedPendulumBulletEnv-v0")
    parser.add_argument("-p", "--population_size", type=int,\
            help="number of individuals in population", default=64)
    parser.add_argument("-w", "--num_workers", type=int,\
            help="number of cpu thread workers", default=0)
    parser.add_argument("-a", "--algorithm", type=str,\
            help="name of es learning algo", default="ESPopulation")
    parser.add_argument("-pi", "--policy", type=str,\
            help="name of policy architecture", default="MLPPolicy")
    parser.add_argument("-g", "--generations", type=int,\
            help="number of generations", default=50)
    parser.add_argument("-t", "--performance_threshold", type=float,\
            help="performance threshold to use for early stopping", default=float("Inf"))
    parser.add_argument("-x", "--exp_name", type=str, \
            help="name of experiment", default="temp_exp")
    parser.add_argument("-s", "--seeds", type=int, nargs="+", default=42,\
            help="seed for initializing pseudo-random number generator")

    args = parser.parse_args()

    if "-v" not in args.env_name:
        args.env_name += "-v0"

    if type(args.seeds) is not list:
        args.seeds = [args.seeds]

    train(args)
    
