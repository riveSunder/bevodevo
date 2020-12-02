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

class ImageMotivator(nn.Module):

    def __init__(self, env_fn, env_name):
        super(ImageMotivator, self).__init__()
        pass

    def reset(self):
        return self.env.reset()

    def forward(self, x):
        pass
                

    def get_loss(self, x):
        pass

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        bonus = self.get_loss(obs) #, self.layers)

        self.update(obs)

        reward_plus = self.curiosity * bonus + (1-self.curiosity) * reward

        info["reward"] = reward
        info["bonus"] =  bonus

        return obs, reward_plus, done, info

    def update(self, obs):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass


class AEImageMotivator(ImageMotivator):

    def __init__(self, env_fn, env_name):
        super(AEImageMotivator, self).__init__(env_fn, env_name)


        self.env = env_fn(env_name)
        self.dim_x = self.env.observation_space.shape
        
        self.dim_ch = np.min(self.dim_x)

        self.conv_filters = 8
        self.kernel_size = 3
        self.curiosity = 0.1
        self.lr = 1e-5

        self.init_ae()

    def init_ae(self):

        self.autoencoder = nn.Sequential(\
                nn.Conv2d(self.dim_ch, self.conv_filters, self.kernel_size,\
                stride=1, padding=1),\
                nn.MaxPool2d(2, 2, 0),\
                nn.Conv2d(self.conv_filters, self.conv_filters, self.kernel_size,\
                stride=1, padding=1),\
                nn.ConvTranspose2d(self.conv_filters, self.conv_filters, self.kernel_size,\
                stride=2, padding=0),
                nn.Conv2d(self.conv_filters, self.dim_ch, 2,\
                stride=1, padding=0)
            )

        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr= self.lr)


    def parse_tensor(self, obs):

        if type(obs) is not torch.Tensor:
            obs = torch.Tensor(obs)

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)

        if obs.shape[-1] <= 4:
            obs = obs.permute(0,3,1,2)

        return obs

    def forward(self, obs):
        
        # obs is an image, which must be permuted to NCHW format
        obs = self.parse_tensor(obs)

        out = self.autoencoder(obs)

        return out

    def get_loss(self, obs):

        obs = self.parse_tensor(obs)

        out = self.forward(obs)
        loss = torch.mean((obs - out)**2)

        return loss

    def update(self, obs):

        loss = self.get_loss(obs)

        loss.backward()

        self.optimizer.step()



class RNDImageMotivator(AEImageMotivator):

    def __init__(self, env_fn, env_name):
        super(RNDImageMotivator, self).__init__(env_fn, env_name)

    def init_ae(self):
        
        self.lr = 1e-7 
        self.curiosity = 0.01
        self.conv_filters = 4
        prod = lambda a,b: a*b
        dense_input = reduce(prod, self.dim_x)

        dense_input = int(dense_input * self.conv_filters/(2**2 * self.dim_ch))
        dense_output = 16

        self.predictor = nn.Sequential(\
                nn.Conv2d(self.dim_ch, self.conv_filters, self.kernel_size,\
                stride=1, padding=1),\
                nn.MaxPool2d(2, 2, 0),\
                nn.Conv2d(self.conv_filters, self.conv_filters, self.kernel_size,\
                stride=1, padding=1),\
                nn.Flatten(),
                nn.Linear(dense_input, dense_output)
            )

        self.random_network = nn.Sequential(\
                nn.Conv2d(self.dim_ch, self.conv_filters, self.kernel_size,\
                stride=1, padding=1),\
                nn.MaxPool2d(2, 2, 0),\
                nn.Conv2d(self.conv_filters, self.conv_filters, self.kernel_size,\
                stride=1, padding=1),\
                nn.Flatten(),
                nn.Linear(dense_input, dense_output)
            )

        for param in self.random_network.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr= self.lr)

    def forward_pred(self, obs):
        # forward pass of prediction model

        obs = self.parse_tensor(obs)

        pred = self.predictor(obs)

        return pred

    def forward_rand(self, obs):

        obs = self.parse_tensor(obs)
        
        out = self.random_network(obs)

        return out

    def get_loss(self, obs):

        pred = self.forward_pred(obs)
        out = self.forward_rand(obs)

        loss = torch.mean((pred - out)**2)

        return loss

if __name__ == "__main__":

    env_name = "Pong-v0"

    for env_wrapper in [RNDImageMotivator, AEImageMotivator]:

        env = env_wrapper(gym.make, env_name)
        rewards = []
        for ii in range(2):

            print("episode {} begin".format(ii))
            obs = env.reset()
            done = False

            while not done:
                obs, reward, done, info = env.step(env.env.action_space.sample())
                #print("reward {:.2e}".format(reward))
                rewards.append(reward)

        
        plt.figure();
        plt.plot(rewards)

    plt.show()
