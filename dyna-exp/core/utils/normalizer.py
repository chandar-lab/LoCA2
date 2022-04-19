#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import torch
from ..utils.tiles_wrapper import TileCoder


class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class TileCodingNormalizer(BaseNormalizer):
    def __init__(self, config):
        BaseNormalizer.__init__(self)
        self.tiles_rep = TileCoder(config)
        self.dim = config.tiles_memsize

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        if x.shape.__len__() == 2:
            y = []
            for i in range(x.shape[0]):
                y.append(self.tiles_rep.get_representation(x[i]))
            y = np.asarray(y)
        else:
            y = self.tiles_rep.get_representation(x)
        return y


class DummyNormalizer(BaseNormalizer):
    def __init__(self, config):
        BaseNormalizer.__init__(self)
        self.dim = config.observation_space.shape[0]

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return x