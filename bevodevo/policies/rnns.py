from collections import OrderedDict
from functools import reduce

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevodevo.policies.base import Policy

class GatedRNNPolicy(Policy):
    """
    This agent only uses autograd and does not depend on PyTorch
    """
    def __init__(self, **kwargs):
        super(GatedRNNPolicy, self).__init__()

        # 
        self.discrete = kwargs["discrete"] if "discrete" in kwargs.keys() else False
        self.use_grad = kwargs["use_grad"] if "use_grad" in kwargs.keys() else False
        self.final_act = nn.Tanh() if not self.discrete else nn.Softmax(dim=-1)

        # architectural meta-parameters
        self.dim_x = kwargs["dim_x"] if "dim_x" in kwargs.keys() else 5 
        self.input_dim = self.dim_x
        self.dim_h = kwargs["dim_h"] if "dim_h" in kwargs.keys() else 16
        self.dim_h = self.dim_h[0] if type(self.dim_h) == list else self.dim_h
        self.dim_y = kwargs["dim_y"] if "dim_y" in kwargs.keys() else 1 
        self.action_dim = self.dim_y
        self.j_act = nn.Tanh()

        # starting parameters for population
        self.var = 0.1**2
        self.init_mean = 0.0

        self.num_params = 2 * (self.dim_x + self.dim_h) * self.dim_h \
                + self.dim_h * self.dim_y
        
        self.init_params()

        if "params" in kwargs.keys() and kwargs["params"] is not None: 
            self.set_params(args["params"])
    
    def init_params(self):

        self.g = nn.Sequential(OrderedDict([\
                ("g", nn.Linear(self.dim_h+self.dim_x, self.dim_h)),\
                ("act_g", nn.Sigmoid())]))

        self.j = nn.Sequential(OrderedDict([\
                ("j", nn.Linear(self.dim_h+self.dim_x, self.dim_h)),\
                ("act_j", nn.Tanh())]))

        self.w_h2y = nn.Sequential(OrderedDict([\
                ("w_h2y", nn.Linear(self.dim_h, self.dim_y)),\
                ("final_act", self.final_act)]))

        self.cell_state = torch.zeros((1,self.dim_h))

        for submodel in [self.g, self.j, self.w_h2y]:
            for param in submodel.parameters():
                param.requires_grad = False

        self.num_params = self.get_params().shape[0]

    def forward(self, x):
        

        x = torch.Tensor(x)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        x = torch.cat((self.cell_state, x), axis=-1)

        g_out = self.g(x) 

        j_out = (1.0 - g_out) * self.j(x)

        self.cell_state = g_out * self.cell_state + j_out

        y = self.w_h2y(self.cell_state) 

        return y
        
    def get_action(self, x):

        y = self.forward(x)

        if self.discrete:
            act = torch.argmax(y, dim=-1)
        else:
            act = y

        return act.detach().cpu().numpy()

    def get_params(self):
        params = np.array([])

        for param in self.g.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        for param in self.j.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        for param in self.w_h2y.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def set_params(self, my_params):

        if my_params is None:
            my_params = self.init_mean + torch.randn(self.num_params) * torch.sqrt(torch.tensor(self.var))

        param_start = 0
        self.init_params()
        for name, param in self.g.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)


            param[:] = torch.nn.Parameter(torch.Tensor(\
                    my_params[param_start:param_stop].reshape(param.shape)))

            param_start = param_stop

        for name, param in self.j.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

            param_start = param_stop


        for name, param in self.w_h2y.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

            param_start = param_stop


        for submodel in [self.g, self.j, self.w_h2y]:
            for param in submodel.parameters():
                param = param.detach() 


    def reset(self):
        self.cell_state *= 0. 
        self.zero_grad()


if __name__ == "__main__":

    # run tests

    print("OK")

