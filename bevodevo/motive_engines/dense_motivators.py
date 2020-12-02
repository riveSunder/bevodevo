from abc import ABC, abstractmethod
import time
from functools import reduce

import autograd.numpy as xnp
from autograd import grad
import numpy as np
 
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import pybullet
import pybullet_envs

import matplotlib.pyplot as plt


class Motivator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self):
        pass

class AEMotivator(Motivator):

    def __init__(self, env_fn, env_name):
        super(AEMotivator, self).__init__()

        self.env = env_fn(env_name)

        self.dim_x = self.env.observation_space.shape[0]

        self.dim_h = [64,64]

        self.init_weights()

        self.get_grad = grad(self.get_loss, argnum=1)
        self.curiosity = 9.9e-1 
        self.lr = 3e-4
        self.mr = 1e-1

    def reset(self):
        self.momentum = [elem * 0.0 for elem in self.layers]
        return self.env.reset()

    def forward(self, x, layers):
        
        for ii in range(len(layers)-1):
            x = xnp.tanh(xnp.matmul(x, layers[ii]))

        return xnp.matmul(x, layers[-1])

    def get_loss(self, obs, layers):

        x_pred = self.forward(obs, layers)

        return xnp.mean(xnp.abs((x_pred - obs)**2))

    def accumulate_momentum(self, my_grad):

        for idx, grads in enumerate(my_grad):
            self.momentum[idx] = (1-self.mr) * self.momentum[idx] + self.mr * grads

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        #bonus = self.get_loss(obs, self.layers)

        my_grad = self.get_grad(obs, self.layers)

        self.accumulate_momentum(my_grad)
        
        for params, mom in zip(self.layers, self.momentum):
            params -= self.lr * mom

        bonus = self.get_loss(obs, self.layers)
        reward_plus = self.curiosity * bonus + (1-self.curiosity) * reward
        
        info["reward"] = reward
        info["bonus"] =  bonus

        return obs, reward_plus, done, info

    def init_weights(self):

        self.layers = []

        self.layers.append(1e-1*np.random.randn(self.dim_x, self.dim_h[0]))

        for kk in range(len(self.dim_h)-1):
            self.layers.append(1e-1 * np.random.randn(self.dim_h[kk], self.dim_h[kk+1]))

        self.layers.append(1e-1 * np.random.randn(self.dim_h[-1], self.dim_x))

    def set_params(self):
        pass

    def get_params(self):
        pass

class RNDMotivator(AEMotivator):
    def __init__(self, env_fn, env_name):
        super(RNDMotivator, self).__init__(env_fn, env_name)

    def get_loss(self, obs, layers):

        rnd_tgt, rnd_pred = self.forward(obs, layers)

        return self.curiosity * xnp.mean(xnp.abs((rnd_pred - rnd_tgt)**2))

    def forward(self, x, layers):
        
        # random network 
        x_rnd = np.tanh(np.tanh(np.matmul(x, self.rnd_layers[0])))

        for hh in range(1, len(self.rnd_layers)-1):
            x_rnd = np.tanh(np.matmul(x_rnd, self.rnd_layers[hh]))

        rnd_tgt = np.matmul(x_rnd, self.rnd_layers[-1])

        # randon network output predictor
        for ii in range(len(layers)-1):
            x = xnp.tanh(xnp.matmul(x, layers[ii]))

        rnd_pred = xnp.matmul(x, layers[-1])

        return rnd_tgt, rnd_pred

    def init_weights(self):

        self.rnd_layers = []

        # get a random network (but the same one each time
        # grab np.random state, leter reset after getting random network deterministically
        npr_state = np.random.get_state()
        np.random.seed(1337)
        self.rnd_layers.append(1e-1*np.random.randn(self.dim_x, self.dim_h[0]))

        for kk in range(len(self.dim_h)-1):
            self.rnd_layers.append(1e-1 * np.random.randn(self.dim_h[kk], self.dim_h[kk+1]))

        self.rnd_layers.append(1e-1 * np.random.randn(self.dim_h[-1], self.dim_x))

        # reset to original state
        np.random.set_state(npr_state)

        self.layers = []

        self.layers.append(1e-1*np.random.randn(self.dim_x, self.dim_h[0]))

        for kk in range(len(self.dim_h)-1):
            self.layers.append(1e-1 * np.random.randn(self.dim_h[kk], self.dim_h[kk+1]))

        self.layers.append(1e-1 * np.random.randn(self.dim_h[-1], self.dim_x))



if __name__ == "__main__":

    env_name = "CartPole-v0"

    for env_wrapper in [RNDMotivator, AEMotivator]:

        env = env_wrapper(gym.make, env_name)
        rewards = []
        for ii in range(256):

            obs = env.reset()
            done = False

            while not done:
                obs, reward, done, info = env.step(env.env.action_space.sample())
                #print("reward {:.2e}".format(reward))
                rewards.append(reward)

        plt.figure();
        plt.plot(rewards)

    plt.show()
