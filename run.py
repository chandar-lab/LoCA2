from core.component.envs import Env
from core.network import network_bodies
from core.network.network_heads import NonLinearDynaQNet
from core import agent
from core.utils import normalizer
from core.utils.big_loop import run_steps
from core.utils.torch_utils import random_seed
from core.utils.param_config import ParamConfig

import torch
import torch.nn as nn
import os
import argparse
from alphaex.sweeper import Sweeper


def set_task_fn(cfg):
    cfg.task_fn = lambda seed: Env(cfg.task_name, cfg, seed=seed, timeout=cfg.timeout)
    cfg.eval_task = cfg.task_fn(seed=0)

    cfg.observation_space = cfg.eval_task.observation_space
    cfg.action_space = cfg.eval_task.action_space


def set_network_fn_and_network(cfg):
    input_dim = cfg.state_normalizer.dim
    action_dim = cfg.action_space.n

    if hasattr(cfg, 'network_name') is False or cfg.network_name is None:
        return
    elif cfg.network_name == 'NonLinearDynaQNet':
        q_body_class = getattr(network_bodies, cfg.q_body)
        s_body_class = getattr(network_bodies, cfg.s_body)
        r_body_class = getattr(network_bodies, cfg.r_body)
        t_body_class = getattr(network_bodies, cfg.t_body)
        cfg.network_fn = lambda: NonLinearDynaQNet(
            input_dim, action_dim,
            nn.ModuleList(
                [s_body_class(
                    input_dim, hidden_units=tuple(cfg.s_body_network),
                    gate=getattr(torch, cfg.s_body_gate)
                ) for _ in range(action_dim)]
            ),
            nn.ModuleList(
                [r_body_class(
                    input_dim, hidden_units=tuple(cfg.r_body_network),
                    gate=getattr(torch, cfg.r_body_gate)
                ) for _ in range(action_dim)]
            ),
            nn.ModuleList(
                [t_body_class(
                    input_dim, hidden_units=tuple(cfg.t_body_network),
                    gate=getattr(torch, cfg.t_body_gate)
                ) for _ in range(action_dim)]
            ),
            q_body_class(
                input_dim, hidden_units=tuple(cfg.q_body_network),
                gate=getattr(torch, cfg.q_body_gate)
            )
        )
    else:
        raise NotImplementedError
    cfg.network = cfg.network_fn()


def set_normalizer_class(cfg):
    normalizer_class = getattr(normalizer, cfg.state_normalizer_name)
    cfg.state_normalizer = normalizer_class(cfg)
    normalizer_class = getattr(normalizer, cfg.reward_normalizer_name)
    cfg.reward_normalizer = normalizer_class(cfg)


def set_agent_class(cfg):
    cfg.agent_class = getattr(agent, cfg.agent_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--id', default=0, type=int, help='identifies run number and configuration')
    parser.add_argument('--config-file')

    args = parser.parse_args()
    exp_name = args.config_file.split('/')[1]
    project_root = os.path.abspath(os.path.dirname(__file__))
    param_sweeper = Sweeper(os.path.join(project_root, args.config_file))
    param_sweeper_dict = param_sweeper.parse(args.id)

    cfg = ParamConfig(param_sweeper_dict)
    cfg.param_sweeper_dict = param_sweeper_dict
    cfg.exp_name = exp_name
    cfg.data_root = os.path.join(project_root, 'data', 'output')
    cfg.sweep_id = args.id

    random_seed(cfg.sweep_id)

    # Setting up the config
    set_task_fn(cfg)

    set_normalizer_class(cfg)

    set_network_fn_and_network(cfg)

    set_agent_class(cfg)

    # make output directory
    bash_script = "mkdir -p experiments/%s/outputs" % cfg.exp_name
    print(bash_script)
    myCmd = os.popen(bash_script).read()

    cfg.rank = 0
    run_steps(cfg)