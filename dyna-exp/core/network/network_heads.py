#######################################################################
# Copyright (C) 2022 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import torch
import torch.nn as nn
from .network_utils import BaseNet, layer_init
from ..utils.torch_utils import tensor
from ..utils.param_config import ParamConfig
from .network_bodies import FCBody


class NonLinearDynaQNet(nn.Module, BaseNet):
    def __init__(
            self, state_dim, action_dim,
            s_body=None, r_body=None, t_body=None, q_body=None
    ):
        super(NonLinearDynaQNet, self).__init__()
        if s_body is None: s_body = FCBody(state_dim, hidden_units=(64, 64))
        if r_body is None: r_body = FCBody(state_dim, hidden_units=(64, 64))
        if t_body is None: t_body = FCBody(state_dim, hidden_units=(64, 64))
        if q_body is None: q_body = FCBody(state_dim, hidden_units=(64, 64))
        
        self.s_body = s_body
        self.r_body = r_body
        self.t_body = t_body
        self.q_body = q_body
        
        self.fc_s = nn.ModuleList([layer_init(nn.Linear(s_body[i].feature_dim, state_dim)) for i in range(action_dim)])
        self.fc_r = nn.ModuleList([layer_init(nn.Linear(r_body[i].feature_dim, 1)) for i in range(action_dim)])
        self.fc_t = nn.ModuleList([layer_init(nn.Linear(t_body[i].feature_dim, 1)) for i in range(action_dim)])
        self.fc_q = layer_init(nn.Linear(q_body.feature_dim, action_dim))

        self.model_params = list(self.s_body.parameters()) + list(self.fc_s.parameters()) + \
                            list(self.r_body.parameters()) + list(self.fc_r.parameters()) + \
                            list(self.t_body.parameters()) + list(self.fc_t.parameters())
        self.value_params = list(self.q_body.parameters()) + list(self.fc_q.parameters())
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.to(ParamConfig.DEVICE)
    
    def forward(self, x, flag):
        state = tensor(x)
        batch_size = x.shape[0]
        if flag == 'q':
            q = self.fc_q(self.q_body(state))
            return q
        elif flag == 'model':
            s_list = []
            r_list = []
            t_list = []
            for action in range(self.action_dim):
                s = state + self.fc_s[action](self.s_body[action](state))
                s_list.append(s)
                r = self.fc_r[action](self.r_body[action](state))
                r_list.append(r)
                t = self.fc_t[action](self.t_body[action](state))
                t_list.append(t)
            s = torch.cat(s_list, dim=1).view(batch_size, self.action_dim, self.state_dim)
            r = torch.cat(r_list, dim=1).view(batch_size, self.action_dim, 1)
            t = torch.cat(t_list, dim=1).view(batch_size, self.action_dim, 1)
            t = torch.sigmoid(t)
            return s, r, t
        else:
            raise NotImplementedError
