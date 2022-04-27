#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch.nn as nn
import torch


class BaseNet:
    def __init__(self):
        pass
    
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


def layer_init(layer, initialization='kaiming_uniform', gate=None):
    if initialization == 'kaiming_uniform':
        if gate is None:
            nn.init.kaiming_uniform_(layer.weight.data, mode='fan_in')
        elif gate.__name__ == 'relu':
            nn.init.kaiming_uniform_(layer.weight.data, mode='fan_in', nonlinearity='relu')
        elif gate.__name__ == 'tanh':
            nn.init.kaiming_uniform_(layer.weight.data, mode='fan_in', nonlinearity='tanh')
        else:
            raise NotImplementedError
    elif initialization == 'all_zeros':
        nn.init.constant_(layer.weight.data, 0)
    elif initialization == 'all_ones':
        nn.init.constant_(layer.weight.data, 1)
    else:
        raise NotImplementedError
    if layer.bias is not None:
        nn.init.constant_(layer.bias.data, 0)
    return layer


def update_weights(optimizer, loss, gradient_clip, parameters):
    optimizer.zero_grad()
    loss.backward()
    if gradient_clip is not None:
        nn.utils.clip_grad_norm_(parameters, gradient_clip)
    optimizer.step()
