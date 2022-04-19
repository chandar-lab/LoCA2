#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


from .network_utils import layer_init
import torch
import torch.nn as nn


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=torch.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out), gate=gate) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.gate(x)
        return x