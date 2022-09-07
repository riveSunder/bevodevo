from collections import OrderedDict
from functools import reduce

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImpalaCNNPolicy(nn.Module):

    def __init__(self, **kwargs):
        super(ImpalaCNNPolicy, self).__init__()

        self.use_grad = kwargs["use_grad"] if "use_grad" in kwargs.keys() else False
        self.input_dim = kwargs["dim_x"] if "dim_x" in kwargs.keys() else 3 
        self.action_dim = kwargs["dim_y"] if "dim_y" in kwargs.keys() else 1


        self.filters = kwargs["filters"] if "filters" in kwargs.keys() else [16, 32, 32]

        self.activations = kwargs["activations"] if "activations" in kwargs.keys() \
                else [F.relu, F.relu, F.relu, F.relu]
                

        # TODO: allow adjustable layers and parameter dims
        if type(self.input_dim) == list:
            assert len(self.input_dim) >= 3, "cnn expects 3D input dims"
        elif type(self.input_dim) == int:
            print("warning, cnns expect 3D input dims. Assuming height, width = 64, 64")
            self.input_dim = [64, 64, args["dim_x"]]


        self.use_bias = True
        self.var = 0.1**2
        self.kern_size = 3


        temp = [elem // 8 + (1 if (elem % 8) else 0) \
                for elem in self.input_dim[0:2]]
        self.hid_input_dim = temp[0]*temp[1] * self.filters[-1]
        
        

        # discrete action space description
        self.discrete = kwargs["discrete"] if "discrete" in kwargs.keys() else False
        self.multilabel = False 
        # discrete output classes are either mutually exclusive (multiclass)
        # or can be combined (multilablel), the latter uses sigmoid and the former softmax

        self.init_params()

        if "params" in kwargs.keys() and kwargs["params"] is not None: 
            self.set_params(kwargs["params"])

        self.num_params = self.get_params().shape
        

    def init_params(self):
        # set up feature extractor
        self.extractor = torch.nn.Sequential(OrderedDict([\
                ("conv_layer0", torch.nn.Conv2d(3, self.filters[0], 3, \
                stride=2, padding=1, bias=False)),\
                ("conv_activation0", torch.nn.ReLU(True))\
                ]))

        for ii in range(1,len(self.filters)):

            self.extractor.add_module("layer{}".format(ii), \
                    torch.nn.Sequential(OrderedDict([\
                    ("conv_layer{}".format(ii), \
                    torch.nn.Conv2d(self.filters[ii-1], self.filters[ii], \
                    3, stride=2, padding=1, bias=False)),\
                    ("conv_activation{}".format(ii), torch.nn.ReLU(True))\
                    ])))

        #assume square input images
        self.top = torch.nn.Sequential(OrderedDict([\
                ("flatten", torch.nn.Flatten()), \
                ("layer0", \
                torch.nn.Linear(self.hid_input_dim, self.filters[0], bias=False)),\
                ("activation0", torch.nn.ReLU(True)), \
                ("layer1", torch.nn.Linear(self.filters[0], self.action_dim, bias=False))\
                ]))

        if self.discrete:

            if self.multilabel:
                self.top.add_module("final_activation",\
                        torch.nn.Sigmoid()
                        )
            else:
                self.top.add_module("final_activation",\
                        torch.nn.Softmax(dim=-1)\
                        )
        else:
            self.top.add_module("final_activation",\
                    torch.nn.Tanh()\
                    )


    def forward(self,x):

        if type(x) is not torch.Tensor:
            x = torch.tensor(x)

        x = x.to(torch.float32)

        if x.shape[-1] == 3:
            x = x.permute(2,0,1)

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = self.extractor(x)
    
        logits = self.top(x)

        return logits

    def get_action(self, x):

        # assuming 8-bit rgb images
        x = x / 255

        logits = self.forward(x)

        if self.discrete:
            action = torch.argmax(logits, dim=-1).detach().numpy()
        else:
            action = logits.detach().numpy()

        return action

    def get_params(self):
        params = np.array([])

        for param in self.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        return params
    

    def set_params(self, my_params):

        param_start = 0
        for name, param in self.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

            param_start = param_stop

    def reset(self):
        pass


if __name__ == "__main__":

    # run tests

    args = {}
    args["dim_x"] = 6
    args["dim_y"] = 1
    args["dim_h"] = 16
    args["params"] = None

    temp = ImpalaCNNPolicy(args)
    x = torch.randn(64,64,3)
    temp_action = temp.get_action(x)
    print("OK")
