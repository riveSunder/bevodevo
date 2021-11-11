from collections import OrderedDict
from functools import reduce

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevodevo.policies.base import Policy

class MLPPolicy(nn.Module):

    def __init__(self, **kwargs):
        super(MLPPolicy, self).__init__()

        self.use_grad = kwargs["use_grad"] if "use_grad" in kwargs.keys() else False

        # architecture params
        self.input_dim = kwargs["dim_x"] if "dim_x" in kwargs.keys() else 5
        self.action_dim = kwargs["dim_y"] if "dim_y" in kwargs.keys() else 1
        self.hid_dims = kwargs["dim_h"] if "dim_h" in kwargs.keys() else 16
        self.hid_dims = [self.hid_dims] if type(self.hid_dims) is not list else self.hid_dims
        self.activations = kwargs["activations"] \
                if "activations" in kwargs.keys() else nn.ReLU
        self.discrete = kwargs["discrete"] if "discrete" in kwargs.keys() else False
        self.use_bias = False
        self.var = 1.e-2

        if type(self.activations) == list:
            if len(self.activations) <= (len(self.hid_dims)+1):
                # use no activation after list of act fns are used up
                for ii in range(len(self.activations), len(self.hid_dims)+1):
                    # identity function for layers missing activations
                    self.activations.append(lambda x: x)
            elif len(self.activations) >= (len(self.hid_dims)+1):
                print("warning: activation list has {} functions but MLP has only {} layers"\
                        .format(len(self.activations), len(self.hid_dims)+1))
                print("... truncating action function list")

                self.activations = self.activations[:len(self.hid_dims)]
        else:
            self.activations = [self.activations] * len(self.hid_dims)

        if self.discrete:
            pass
            # TODO: test/implement discrete action spaces
            #self.activations.append(lambda x: x)
        else:
            self.activations.append(nn.Tanh)

        self.init_params()

        if kwargs["params"] is not None: 
            self.set_params(kwargs["params"])

    def init_params(self):

        self.layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))

        for jj in range(1, len(self.hid_dims)):
            self.layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))
            self.layers.add_module("activation{}".format(jj), self.activations[jj]())

        self.layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))
        if self.discrete:
            pass
        else:
            self.layers.add_module("output_activation",\
                    self.activations[-1]())

        for param in self.layers.parameters():
            param.requires_grad = self.use_grad

        self.num_params = self.get_params().shape[0]

    def forward(self, x):

        if type(x) is not torch.Tensor:
            x = torch.tensor(x)

        x = x.to(torch.float32)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.layers(x)
        return x

    def get_action(self, x):

        y = self.forward(x)

        if self.discrete:
            act = torch.argmax(y, dim=-1)
        else:
            act = y

        return act.detach().cpu().numpy()

    def get_params(self):
        params = np.array([])

        for param in self.layers.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def set_params(self, my_params):

        param_start = 0
        for name, param in self.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

    def reset(self):
        pass


class HebbianMLP(MLPPolicy):

    def __init__(self, **kwargs):
        self.plastic = kwargs["plastic"] if "plastic" in kwargs.keys() else True
        self.lr_layers = None
        self.e_min = -1.
        self.e_max = 1.
        
        super(HebbianMLP, self).__init__(**kwargs)
        if self.plastic:
            self.init_traces()
        else:
            self.clear_nodes()

    def init_traces(self):

        self.dim_list = [self.input_dim]
        self.dim_list.extend(self.hid_dims)
        self.dim_list.append(self.action_dim)

        # clear node activations, start at 0 everywhere
        self.clear_nodes()

        # initialize learning rate values
        self.lr_layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))

        self.eligibility_layers = [torch.zeros(self.input_dim, self.hid_dims[0])]

        for jj in range(1, len(self.hid_dims)-1):
            self.lr_layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))
            self.lr_layers.add_module("activation{}".format(jj), self.activations[jj]())

            self.eligibility_layers.append(torch.zeros(self.hid_dims[jj], self.hid_dims[jj+1]))

        self.lr_layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))

        self.eligibility_layers.append(torch.zeros(self.hid_dims[-1], self.action_dim))

        if self.discrete:
            pass
        else:
            self.lr_layers.add_module("output_activation",\
                    self.activations[-1]())

        for param in self.lr_layers.parameters():
            param.requires_grad = self.use_grad

        self.num_params = self.get_params().shape[0]


    def clear_nodes(self):

        self.nodes = [torch.zeros(1,elem) for elem in self.dim_list]

    def clear_traces(self):

        if self.plastic:
            self.eligibility_layers = [0.0 * elem for elem in self.eligibility_layers]

    def forward(self, x):

        if type(x) is not torch.Tensor:
            x = torch.tensor(x)

        x = x.to(torch.float32)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        trace_count = 0
        self.nodes[trace_count] = x.clone()

        for name, module in self.layers.named_modules():

            if "layer" in name:

                trace_count += 1
                x = module(x)
                self.nodes[trace_count] = x.clone()
            elif "activation" in name:
                x = module(x)

        if self.plastic:
            self.update()

        return x

    def update(self):

        num_layers = len(list(self.layers.named_parameters()))
        layer_count = 0

        for lr_param, param in zip(list(self.lr_layers.named_parameters()), \
                list(self.layers.named_parameters())):

            layer_dim_x, layer_dim_y = param[1].shape[1], param[1].shape[0]

            self.eligibility_layers[layer_count] += torch.matmul(self.nodes[layer_count].T, \
                    self.nodes[layer_count+1]) 

            self.eligibility_layers[layer_count] = torch.clamp(self.eligibility_layers[layer_count], min=self.e_min, max=self.e_max)

            for ii in range(layer_dim_x):
                for jj in range(layer_dim_y):

                    param[1][jj,ii] = param[1][jj,ii] + lr_param[1][jj,ii] * self.eligibility_layers[layer_count][ii,jj] 

            layer_count += 1

    def get_params(self):
        params = np.array([])

        for param in self.layers.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        if self.lr_layers is not None and self.plastic:
            for param in self.lr_layers.named_parameters():
                params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def set_params(self, my_params):

        param_start = 0
        for name, param in self.layers.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

        if self.plastic:
            for name, param in self.lr_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)

    def reset(self):

        self.clear_nodes()
        self.clear_traces()

class ABCHebbianMLP(HebbianMLP):

    def __init__(self, **kwargs):

        super(ABCHebbianMLP, self).__init__(**kwargs)

    def init_traces(self):

        # clear node activations, start at 0 everywhere
        self.clear_nodes()

        # learning rules are encoded by lr, A, B, C. 
        # \delta W_{ij} = lr * (A * o_i*o_j + B * o_i + C * o_j)
        # initialize learning rate values
        self.lr_layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))


        # Hebbian coefficient A
        self.a_layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))

        # pre-synaptic coefficient B
        self.b_layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))

        # post-synaptic coefficient C
        self.c_layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))

        self.eligibility_layers = [torch.zeros(self.input_dim, self.hid_dims[0])]

        for jj in range(1, len(self.hid_dims)-1):
            self.lr_layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))

            self.a_layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))
            self.b_layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))
            self.c_layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))

            self.eligibility_layers.append(torch.zeros(self.hid_dims[jj], self.hid_dims[jj+1]))

        self.lr_layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))

        self.a_layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))
        self.b_layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))
        self.c_layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))

        self.eligibility_layers.append(torch.zeros(self.hid_dims[-1], self.action_dim))


        for params in [self.lr_layers.parameters(), self.a_layers.parameters(), \
                self.b_layers.parameters(), self.c_layers.parameters()]:
            for param in params: 
                param.requires_grad = self.use_grad

        self.num_params = self.get_params().shape[0]
        

    def update(self):

        num_layers = len(list(self.layers.named_parameters()))
        layer_count = 0

        for lr_param, param, A, B, C in zip(\
                list(self.lr_layers.named_parameters()),\
                list(self.layers.named_parameters()),\
                list(self.a_layers.named_parameters()),\
                list(self.b_layers.named_parameters()),\
                list(self.c_layers.named_parameters())):

            layer_dim_x, layer_dim_y = param[1].shape[1], param[1].shape[0]

            self.eligibility_layers[layer_count] += torch.matmul(self.nodes[layer_count].T, self.nodes[layer_count+1]) 

            self.eligibility_layers[layer_count] = torch.clamp(self.eligibility_layers[layer_count], min=self.e_min, max=self.e_max)

            for ii in range(layer_dim_x):
                for jj in range(layer_dim_y):

                    param[1][jj,ii] = torch.clamp(param[1][jj,ii] + lr_param[1][jj,ii] \
                            * (\
                                    A[1][jj,ii] * self.eligibility_layers[layer_count][ii,jj] \
                                    +B[1][jj,ii] * self.nodes[layer_count][:,ii] \
                                    +C[1][jj,ii] * self.nodes[layer_count+1][:,jj] \
                                ), min=-10, max=10)
                            

            layer_count += 1

    def get_params(self):
        params = np.array([])

        for param in self.layers.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        if self.lr_layers is not None and self.plastic:
            for param in self.lr_layers.named_parameters():
                params = np.append(params, param[1].detach().numpy().ravel())
            for param in self.a_layers.named_parameters():
                params = np.append(params, param[1].detach().numpy().ravel())
            for param in self.b_layers.named_parameters():
                params = np.append(params, param[1].detach().numpy().ravel())
            for param in self.c_layers.named_parameters():
                params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def set_params(self, my_params):

        param_start = 0
        for name, param in self.layers.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

        if self.plastic:
            for name, param in self.lr_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)

            for name, param in self.a_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)
            for name, param in self.b_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)
            for name, param in self.c_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)


class HebbianCAMLP(HebbianMLP):

    def __init__(self, **kwargs):
        super(HebbianCAMLP, self).__init__(**kwargs)
        self.init_pgen()
        
        # Retaining these comments for now, as I might implement the described idea later.
        # two extra outputs for the number of update steps to use when applying CA rules 
        # and the probability of any given cell being changed. (Cells are weights in this policy)
        # self.action_dim += 2

        self.init_traces()
        self.ca_steps = 8
        
    def init_params(self):

        self.layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))

        for jj in range(1, len(self.hid_dims)):
            self.layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))
            self.layers.add_module("activation{}".format(jj), self.activations[jj]())

        self.layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))
        if self.discrete:
            pass
        else:
            self.layers.add_module("output_activation",\
                    self.activations[-1]())

        for param in self.layers.parameters():
            param.requires_grad = self.use_grad

        self.init_pgen()
        self.num_params = self.get_params().shape[0]

    def init_traces(self):

        self.dim_list = [self.input_dim]
        self.dim_list.extend(self.hid_dims)
        self.dim_list.append(self.action_dim)
        self.eligibility_layers = [torch.zeros(self.hid_dims[0], self.input_dim)]

        for jj in range(len(self.hid_dims)-1):

            self.eligibility_layers.append(torch.zeros(self.hid_dims[jj+1], self.hid_dims[jj]))

        self.eligibility_layers.append(torch.zeros(self.action_dim, self.hid_dims[-1]))

        self.init_pgen()

    def init_pgen(self):
        # Policy generating network is comprised of a 2 layer MLP, but implemented as convolutional layers 

        self.pgen = []

        for mm in range(len(self.hid_dims)+2):
            self.pgen.append(nn.Sequential(\
                    nn.Conv2d(4,16,1, stride=1, padding=0, bias=False),\
                    nn.Tanh(),\
                    nn.Conv2d(16,4,1, stride=1, padding=0, bias=False),\
                    nn.Tanh()))

    def update(self):

        num_layers = len(list(self.layers.named_parameters()))
        layer_count = 0

        dim_ch = 4

        for param in list(self.layers.named_parameters()):

            layer_dim_x, layer_dim_y = param[1].shape[1], param[1].shape[0]
            state_grid = torch.ones(1, dim_ch, layer_dim_y, layer_dim_x, requires_grad=False)

            self.eligibility_layers[layer_count] += torch.matmul(self.nodes[layer_count+1].T, self.nodes[layer_count]) 
            self.eligibility_layers[layer_count] = torch.clamp(self.eligibility_layers[layer_count], min=self.e_min, max=self.e_max)

            state_grid[0,0,:,:] = param[1]
            state_grid[0,1,:,:] = self.eligibility_layers[layer_count]
            state_grid[0,2,:,:] *= self.nodes[layer_count]
            state_grid[0,3,:,:] *= self.nodes[layer_count + 1].T
            
            param[1][:,:] = self.pgen[layer_count](state_grid)[0,0]#.squeeze()

            layer_count += 1

    def get_params(self):
        params = np.array([])

        #for param in self.layers.named_parameters():
        #    params = np.append(params, param[1].detach().numpy().ravel())

        if self.plastic:
            for ll in range(len(self.pgen)):
                for param in self.pgen[ll].named_parameters():
                    params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def set_params(self, my_params):

        param_start = 0

        for model in self.pgen:
            for name, param in model.named_parameters():
                
                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)

class HebbianCAMLP2(HebbianCAMLP):

    def __init__(self, **kwargs):
        super(HebbianCAMLP2, self).__init__(**kwargs)
        
        # Retaining these comments for now, as I might implement the described idea later.
        # two extra outputs for the number of update steps to use when applying CA rules 
        # and the probability of any given cell being changed. (Cells are weights in this policy)
        # self.action_dim += 2

        self.ca_steps = 4
        

    def init_pgen(self):
        # Policy generating network is comprised of a 2 layer MLP, but implemented as convolutional layers 

        self.pgen = []

        for mm in range(len(self.hid_dims)+2):
            self.pgen.append(nn.Sequential(\
                    nn.Conv2d(16,16,1, stride=1, padding=0, bias=False),\
                    nn.Tanh(),\
                    nn.Conv2d(16,4,1, stride=1, padding=0, bias=False),\
                    nn.Tanh()))

    def neighborhood(self, state_grid):

        my_dim = state_grid.shape[1]
        moore = torch.tensor(np.array([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]]),\
                dtype=torch.float32)
        sobel_y = torch.tensor(np.array([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]),\
                dtype=torch.float32)
        sobel_x = torch.tensor(np.array([[[[-1, 2, -1], [0, 0, 0], [1, 2, 1]]]]), \
                dtype=torch.float32)

        moore /= torch.sum(moore)

        sobel_x = sobel_x * torch.ones((state_grid.shape[1], 1, 3,3))
        sobel_y = sobel_y * torch.ones((state_grid.shape[1], 1, 3,3))
        #sobel_x = sobel_x.to(device)
        #sobel_y = sobel_y.to(device)

        moore = moore * torch.ones((state_grid.shape[1], 1, 3,3))
        #moore = moore.to(device)

        grad_x = F.conv2d(state_grid, sobel_x, padding=1, groups=my_dim)
        grad_y = F.conv2d(state_grid, sobel_y, padding=1, groups=my_dim)

        moore_neighborhood = F.conv2d(state_grid, moore, padding=1, groups=my_dim)

        perception = torch.cat([state_grid, moore_neighborhood, grad_x, grad_y], axis=1)

        return perception

    def update(self):

        num_layers = len(list(self.layers.named_parameters()))
        layer_count = 0

        dim_ch = 4

        for param in list(self.layers.named_parameters()):

            layer_dim_x, layer_dim_y = param[1].shape[1], param[1].shape[0]
            state_grid = torch.ones(1, dim_ch, layer_dim_y, layer_dim_x, requires_grad=False)

            self.eligibility_layers[layer_count] += torch.matmul(self.nodes[layer_count+1].T, self.nodes[layer_count]) 
            self.eligibility_layers[layer_count] = torch.clamp(self.eligibility_layers[layer_count], min=self.e_min, max=self.e_max)

            state_grid[0,0,:,:] = param[1]
            state_grid[0,1,:,:] = self.eligibility_layers[layer_count]
            state_grid[0,2,:,:] *= self.nodes[layer_count]
            state_grid[0,3,:,:] *= self.nodes[layer_count + 1].T

            for jj in range(self.ca_steps):
                perception = self.neighborhood(state_grid[:,:4])
                state_grid = self.pgen[layer_count](perception)


            param[1][:,:] = state_grid[0, 0, :, :].squeeze()

            layer_count += 1


class CPPNHebbianMLP(HebbianMLP):

    def __init__(self, **kwargs): 

        super(CPPNHebbianMLP, self).__init__(**kwargs)
        self.plastic = False
        self.init_traces()


        self.init_cppn()
        self.set_params(self.get_cppn_params())

    def init_cppn(self):

        self.cppn_in = 6
        self.cppn_out = 2
        self.cppn_h = [32]
        self.cppn_act = [nn.LeakyReLU, lambda x: x]
        self.cppn_out_act = [nn.Tanh, nn.Sigmoid]

        
        self.cppn = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.cppn_in, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.cppn_act[0]())\
                ]))

        for jj in range(1, len(self.cppn_h)-1):
            self.cppn.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))
            self.cppn.add_module("activation{}".format(jj), self.cppn_act[jj]())

        self.cppn.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.cppn_out, bias=self.use_bias))


        for param in self.cppn.parameters():
            param.requires_grad = self.use_grad

        self.num_params = self.get_cppn_params().shape[0]

    def build_mlp(self):

        num_layers = len(list(self.layers.named_parameters()))
        trace_count = 0
        for layer_num, param in enumerate(list(self.layers.named_parameters())):

            layer_dim_x, layer_dim_y = param[1].shape[1], param[1].shape[0]
            for ii in range(layer_dim_x):
                for jj in range(layer_dim_y):
                    
                    cppn_input = torch.Tensor([layer_num/num_layers - 0.5,\
                            ii/layer_dim_x - 0.5, \
                            jj/ layer_dim_y - 0.5,\
                            self.nodes[trace_count][0,ii],\
                            self.nodes[trace_count+1][0,jj],\
                            param[1][jj,ii]\
                            ])\
                            .unsqueeze(0)

                    weight = self.cppn.forward(cppn_input)
                    param[1][jj,ii] = torch.tanh(weight[:,0]) * torch.sigmoid(weight[:,1])

            trace_count += 1


    def get_action(self, x):

        self.build_mlp()

        y = self.forward(x)

        if self.discrete:
            act = torch.argmax(y, dim=-1)
        else:
            act = y

        return act.detach().cpu().numpy()


    def set_params(self, my_params):

        # set the cppn params, which are then used to set the mlp params
        param_start = 0

        for name, param in self.cppn.named_parameters():
            
            param.requires_grad = False
            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)


        # 
        self.build_mlp()

    def get_cppn_params(self):
        params = np.array([])

        for param in self.cppn.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def get_params(self):
        params = np.array([])

        for param in self.layers.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        return params

class CPPNMLPPolicy(MLPPolicy):

    def __init__(self, **kwargs):
        super(CPPNMLPPolicy, self).__init__(**kwargs)
        """
        CPPN

        input (3) by dim_h (32) by output (2) 
        input is the layer + weight coordinate for each weight
        output is a gating function and weight strength, the product of these ouputs defines weight values

        The CPPN defines the MLP, which is the actually policy

        dim_x () by dim_hp (64) by dim_y 
        dim_x and dim_y are the observation space and action space dimensions.
        """

        self.use_bias = False
        self.init_cppn()
        self.set_params(self.get_cppn_params())

    def init_cppn(self):

        self.cppn_in = 3
        self.cppn_out = 2
        self.cppn_h = [32]
        self.cppn_act = [nn.LeakyReLU, lambda x: x]
        self.cppn_out_act = [nn.Tanh, nn.Sigmoid]
        
        self.cppn = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.cppn_in, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.cppn_act[0]())\
                ]))

        for jj in range(1, len(self.cppn_h)-1):
            self.cppn.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))
            self.cppn.add_module("activation{}".format(jj), self.cppn_act[jj]())

        self.cppn.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.cppn_out, bias=self.use_bias))


        for param in self.cppn.parameters():
            param.requires_grad = self.use_grad

        self.num_params = self.get_cppn_params().shape[0]

    def build_mlp(self):

        num_layers = len(list(self.layers.named_parameters()))
        for layer_num, param in enumerate(list(self.layers.named_parameters())):

            layer_dim_x, layer_dim_y = param[1].shape[1], param[1].shape[0]
            for ii in range(layer_dim_x):
                for jj in range(layer_dim_y):
                    
                    cppn_input = torch.Tensor([layer_num/num_layers - 0.5,\
                            ii/layer_dim_x - 0.5, \
                            jj/ layer_dim_y - 0.5])\
                            .unsqueeze(0)
                    weight = self.cppn.forward(cppn_input)
                    param[1][jj,ii] = torch.tanh(weight[:,0]) * torch.sigmoid(weight[:,1])


    def set_params(self, my_params):

        # set the cppn params, which are then used to set the mlp params
        param_start = 0
        for name, param in self.cppn.named_parameters():
            
            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

        # 
        self.build_mlp()

    def get_cppn_params(self):
        params = np.array([])

        for param in self.cppn.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def get_params(self):
        params = np.array([])

        for param in self.layers.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        return params


if __name__ == "__main__":

    # run tests

    args = {}
    args["dim_x"] = 6
    args["dim_y"] = 1
    args["dim_h"] = 16
    args["params"] = None

    temp = MLPPolicy(args)
    temp = HebbianMLP(args)
    temp = ABCHebbianMLP(args)
    temp = HebbianCAMLP(args)
    temp = CPPNHebbianMLP(args)
    temp = CPPNMLPPolicy(args)

    print("OK")
