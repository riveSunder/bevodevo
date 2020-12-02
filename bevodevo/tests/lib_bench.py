"""

** Matrix Multiply/ Dense Layers
(JAX uses GPU where possible)

env InvertedPendulumBulletEnv-v0 completed 10000 steps in 14.084837436676025 s by  <class '__main__.JAXDummy'>
env InvertedPendulumBulletEnv-v0 completed 10000 steps in 0.9784562587738037 s by  <class '__main__.TorchDummy'>
env InvertedPendulumBulletEnv-v0 completed 10000 steps in 1.3700430393218994 s by  <class '__main__.TorchNNDummy'>
env InvertedPendulumBulletEnv-v0 completed 10000 steps in 0.6532621383666992 s by  <class '__main__.AutogradDummy'>
env InvertedPendulumBulletEnv-v0 completed 10000 steps in 0.6000583171844482 s by  <class '__main__.NPDummy'>
env InvertedDoublePendulumBulletEnv-v0 completed 10000 steps in 13.085666418075562 s by  <class '__main__.JAXDummy'>
env InvertedDoublePendulumBulletEnv-v0 completed 10000 steps in 1.1041879653930664 s by  <class '__main__.TorchDummy'>
env InvertedDoublePendulumBulletEnv-v0 completed 10000 steps in 1.4902136325836182 s by  <class '__main__.TorchNNDummy'>
env InvertedDoublePendulumBulletEnv-v0 completed 10000 steps in 0.7666647434234619 s by  <class '__main__.AutogradDummy'>
env InvertedDoublePendulumBulletEnv-v0 completed 10000 steps in 0.6752419471740723 s by  <class '__main__.NPDummy'>
env AntBulletEnv-v0 completed 10000 steps in 74.12556529045105 s by  <class '__main__.JAXDummy'>
env AntBulletEnv-v0 completed 10000 steps in 6.913511753082275 s by  <class '__main__.TorchDummy'>
env AntBulletEnv-v0 completed 10000 steps in 6.9401490688323975 s by  <class '__main__.TorchNNDummy'>
env AntBulletEnv-v0 completed 10000 steps in 6.7208571434021 s by  <class '__main__.AutogradDummy'>
env AntBulletEnv-v0 completed 10000 steps in 6.381152153015137 s by  <class '__main__.NPDummy'>
env HumanoidBulletEnv-v0 completed 10000 steps in 146.7232747077942 s by  <class '__main__.JAXDummy'>
env HumanoidBulletEnv-v0 completed 10000 steps in 10.63016414642334 s by  <class '__main__.TorchDummy'>
env HumanoidBulletEnv-v0 completed 10000 steps in 10.306998252868652 s by  <class '__main__.TorchNNDummy'>
env HumanoidBulletEnv-v0 completed 10000 steps in 9.857022047042847 s by  <class '__main__.AutogradDummy'>
env HumanoidBulletEnv-v0 completed 10000 steps in 9.750605344772339 s by  <class '__main__.NPDummy'>

** Convolutions **
(JAX and TorchConvGPU use GPU where possible)

env Breakout-v0 completed 10000 steps in 43.28060865402222 s by  <class '__main__.JAXConvDummy'>
env Breakout-v0 completed 10000 steps in 18.163710117340088 s by  <class '__main__.TorchConvGPUDummy'>
env Breakout-v0 completed 10000 steps in 70.16366362571716 s by  <class '__main__.TorchConvDummy'>
env Breakout-v0 completed 500 steps in 901.5561814308167 s by  <class '__main__.AutogradConvDummy'>
env procgen:procgen-coinrun-v0 completed 10000 steps in 35.86895728111267 s by  <class '__main__.JAXConvDummy'>
env procgen:procgen-coinrun-v0 completed 10000 steps in 4.494136571884155 s by  <class '__main__.TorchConvGPUDummy'>
env procgen:procgen-coinrun-v0 completed 10000 steps in 12.05394196510315 s by  <class '__main__.TorchConvDummy'>
env procgen:procgen-coinrun-v0 completed 500 steps in 127.6902346611023 s by  <class '__main__.AutogradConvDummy'>

"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import reduce
import time

from autograd import numpy as anp
import autograd.scipy.signal
convolve = autograd.scipy.signal.convolve

from jax import numpy as jnp
from jax import jit, random
from jax.experimental import stax, optimizers
from jax.experimental.stax import GeneralConv, Dense, LogSoftmax, Flatten

import numpy.random as npr

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import pybullet_envs
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + anp.exp(-x))

def relu(x):
    return x 


class NPDummy():

    def __init__(self, input_dim, output_dim, hid_dim=[32,32]):

        self.dim_x = input_dim
        self.dim_y = output_dim
        self.dim_h = hid_dim

        self.act = [relu] * len(self.dim_h)
        self.act.append(jnp.tanh)

        self.init_parameters()

    def init_parameters(self):
        
        self.layers = []

        self.layers.append(1.e-1 * npr.randn(self.dim_x, self.dim_h[0]))

        for ii in range(1,len(self.dim_h)-1):
            self.layers.append(1.e-1 * npr.randn(self.dim_h[ii-1], self.dim_h[ii]))

        
        self.layers.append(1.e-1 * npr.randn(self.dim_h[-1], self.dim_y))

    def forward(self,x):

        return npr.randn(self.dim_y,)

class JAXDummy(NPDummy):

    def __init__(self, input_dim, output_dim, hid_dim=[32,32]):
        super(JAXDummy, self).__init__(input_dim, output_dim, hid_dim)

        self.act = [jnp.tanh] * len(self.dim_h)
        self.act.append(jnp.tanh)


    #@jit
    def forward(self,x):
        
        for jj, layer in enumerate(self.layers):
            x = self.act[jj](jnp.matmul(x, layer))

        return x


class AutogradDummy(NPDummy):

    def __init__(self, input_dim, output_dim, hid_dim=[32,32]):
        super(AutogradDummy, self).__init__(input_dim, output_dim, hid_dim)

        self.act = [anp.tanh] * len(self.dim_h)
        self.act.append(anp.tanh)

    def forward(self,x):
        
        for jj, layer in enumerate(self.layers):
            x = self.act[jj](anp.matmul(x, layer))

        return x

class TorchDummy(NPDummy):

    def __init__(self, input_dim, output_dim, hid_dim=[32,32]):
        super(TorchDummy, self).__init__(input_dim, output_dim, hid_dim)

        self.act = [torch.tanh] * len(self.dim_h)
    
        self.act.append(torch.tanh)

    def init_parameters(self):
        
        self.layers = []

        self.layers.append(1.e-1 * torch.randn(self.dim_x, self.dim_h[0]))

        for ii in range(1,len(self.dim_h)-1):
            self.layers.append(1.e-1 * torch.randn(self.dim_h[ii-1], self.dim_h[ii]))

        
        self.layers.append(1.e-1 * torch.randn(self.dim_h[-1], self.dim_y))

    def forward(self,x):
        
        x = torch.Tensor(x)

        for jj, layer in enumerate(self.layers):
            x = self.act[jj](torch.matmul(x, layer))

        return x.detach().numpy()

class TorchNNDummy(NPDummy):

    def __init__(self, input_dim, output_dim, hid_dim=[32,32]):
        super(TorchNNDummy, self).__init__(input_dim, output_dim, hid_dim)

        self.act = [torch.tanh] * len(self.dim_h)
    
        self.act.append(torch.tanh)

    def init_parameters(self):
        
        self.layers = nn.Sequential(nn.Linear(self.dim_x, self.dim_h[0], bias=False),\
                nn.Tanh())

        for ii in range(1,len(self.dim_h)-1):
            self.layers.add_module("layer{}".format(ii), \
                    nn.Linear(self.dim_h[ii-1], self.dim_h[ii], bias=False))
            self.layers.add_module("act{}".format(ii),\
                    nn.Tanh())

        
        self.layers.add_module("endlayer",\
                nn.Linear(self.dim_h[-1], self.dim_y, bias=False))
        self.layers.add_module("endact",\
                nn.Tanh())

    def forward(self,x):
        x = torch.Tensor(x)
        return self.layers(x).detach().numpy()

class TorchConvDummy():

    def __init__(self, input_dim, output_dim, hid_dim=[32,32,32,32]):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim_h = hid_dim

        self.init_parameters()

    def init_parameters(self):
        
        self.layers = nn.Sequential(nn.Conv2d(3, self.dim_h[0], 3, stride=1, \
                padding=0,  bias=False), nn.Tanh())

        for ii in range(1,len(self.dim_h)):
            self.layers.add_module("layer{}".format(ii), \
                    nn.Conv2d(self.dim_h[ii-1], self.dim_h[ii], 3, stride=1, \
                    padding=0, bias=False))
            self.layers.add_module("act{}".format(ii),\
                    nn.Tanh())

        self.layers.add_module("flattener", nn.Flatten())

        flat_size = (self.input_dim[0] - 8) * (self.input_dim[1] - 8) * 32

        self.layers.add_module("endlayer",\
                nn.Linear(flat_size, self.output_dim, bias=False))
        self.layers.add_module("endact",\
                nn.Softmax(dim=-1))

    def forward(self,x):
        x = torch.Tensor(x).permute(2,0,1).unsqueeze(0)
        return self.layers(x)

    def get_action(self,x):
        x = self.forward(x)
        action = torch.argmax(x)
        return action.detach().numpy()

class TorchConvGPUDummy(TorchConvDummy):

    def __init__(self, input_dim, output_dim, hid_dim=[32,32,32,32]):
        super(TorchConvGPUDummy, self).__init__(input_dim, output_dim, hid_dim)

        self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

        self.layers.to(self.device)

    def forward(self,x):
        x = torch.Tensor(x).permute(2,0,1).unsqueeze(0).to(self.device)
        y = self.layers(x)
        return y

    def get_action(self,x):
        x = self.forward(x)
        action = torch.argmax(x)
        return action.detach().cpu().numpy()


class AutogradConvDummy():

    def __init__(self, input_dim, output_dim, hid_dim=[32,32,32,32]):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim_h = hid_dim

        self.init_parameters()

    def init_parameters(self):

        self.kernels = []

        self.kernels.append(1e-1 * npr.randn(3, self.dim_h[0], 3, 3))

        for ii in range(1, len(self.dim_h)):

            self.kernels.append(1e-1 * npr.randn(\
                    self.dim_h[ii-1], self.dim_h[ii], 3, 3))

        flat_size = (self.input_dim[0] - 8) * (self.input_dim[1] - 8) * 32
        self.dense = 1e-1 * npr.randn(flat_size, self.output_dim)

    def softmax(self, x):
        x = x - anp.max(x)
        return anp.exp(x) / anp.sum(anp.exp(x), axis=-1)

    def forward(self, x):

        x = x.transpose(2,0,1)[anp.newaxis,:,:,:]

        for jj, kernel in enumerate(self.kernels):

            x = anp.tanh(convolve(x, kernel, axes=([2,3], [2,3]), \
                    dot_axes=([1], [0]), mode="valid"))

        x = x.ravel()

        x = self.softmax(anp.matmul(x, self.dense))

        return x
    
    def get_action(self, x):

        x = self.forward(x)
        action = anp.argmax(x, axis=-1)

        return action


class JAXConvDummy():

    def __init__(self, input_dim, output_dim, hid_dim=[33,33,33,33]):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim_h = hid_dim

        self.init_parameters()
        

    def init_parameters(self):

        rng_key = random.PRNGKey(0)

        self.init_fn, self.pred_fn = stax.serial(\
                GeneralConv(("NCHW", "IOHW", "NCHW"), self.dim_h[0], (3,3), (1,1), "VALID"),\
                GeneralConv(("NCHW", "IOHW", "NCHW"), self.dim_h[1], (3,3), (1,1), "VALID"),\
                GeneralConv(("NCHW", "IOHW", "NCHW"), self.dim_h[2], (3,3), (1,1), "VALID"),\
                GeneralConv(("NCHW", "IOHW", "NCHW"), self.dim_h[3], (3,3), (1,1), "VALID"),\
                Flatten,\
                Dense(self.output_dim),\
                LogSoftmax)

        _, self.init_params = self.init_fn(rng_key, (1, 3, self.input_dim[0], self.input_dim[1]))

        self.opt_init, self.opt_update, self.get_params = optimizers.momentum(1e-3, mass=0.9)
        self.opt_state = self.opt_init(self.init_params)

        self.params = self.get_params(self.opt_state)

    def forward(self, x):

        x = 1.0 * x.transpose(2,0,1)[jnp.newaxis,:,:,:]

        
        x = self.pred_fn(self.params, x)

        return x
    
    def get_action(self, x):

        x = self.forward(x)
        action = jnp.argmax(x, axis=-1)

        return action[0]


if __name__ == "__main__":

    max_steps = 10000

    if(1):
        for env_name in ["InvertedPendulumBulletEnv-v0",\
                "InvertedDoublePendulumBulletEnv-v0",\
                "AntBulletEnv-v0",\
                "HumanoidBulletEnv-v0"]:

            env = gym.make(env_name)
            input_dim = env.observation_space.sample().shape[0]
            output_dim = env.action_space.sample().shape[0]

            for agent_fn in [JAXDummy, TorchDummy, TorchNNDummy, AutogradDummy, NPDummy]:

                steps = 0

                model = agent_fn(input_dim=input_dim, output_dim=output_dim, hid_dim=[32,32])

                t0 = time.time()
                done = True
                while steps < max_steps:
                    if done:
                        obs = env.reset()
                        done = False
                    
                    obs, reward, done, info = env.step(model.forward(obs))
                    steps += 1

                t1 = time.time()
                print("env {} completed {} steps in {} s by ".format(env_name, steps, t1-t0), agent_fn)

            env.close()


    for env_name in [ "Breakout-v0", "procgen:procgen-coinrun-v0"]:

        env = gym.make(env_name)
        input_dim = env.observation_space.sample().shape
        output_dim = env.action_space.n

        for agent_fn in [JAXConvDummy, TorchConvGPUDummy, TorchConvDummy, AutogradConvDummy ]:

            if agent_fn == AutogradConvDummy:
                max_steps = 500
            else:
                max_steps = 10000

            steps = 0
            model = agent_fn(input_dim=input_dim, output_dim=output_dim)

            t0 = time.time()

            done = True
            while steps < max_steps:
                if done:
                    obs = env.reset()
                    done = False
                
                action = model.get_action(obs)
                obs, reward, done, info = env.step(action)
                steps += 1

            t1 = time.time()
            print("env {} completed {} steps in {} s by ".format(env_name, steps, t1-t0), agent_fn)

        env.close()
