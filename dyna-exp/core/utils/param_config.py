#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from .misc import ensure_dir
import argparse
import torch
import os
import datetime


class ParamConfig:
    DEVICE = torch.device('cpu')

    def __init__(self, param_dict):
        self.parser = argparse.ArgumentParser()
        
        # the following things must be defined in config file
        self.exp_name = None
        self.task_name = None
        self.task_args = None
        self.agent_name = None
        self.network_name = None
        
        # problem definition
        self.discount = 0.99
        self.state_normalizer_name = 'DummyNormalizer'
        self.reward_normalizer_name = 'DummyNormalizer'
        self.history_length = None
        
        # experimental parameters
        self.log_level = 0
        self.log_interval = 1000
        self.if_save_network = False
        self.save_interval = 100000
        self.if_eval_steps = False
        self.if_eval_episodes = False
        self.eval_interval = 5000
        self.eval_steps = 1000
        self.eval_episodes = 100
        self.max_steps = 1000000
        self.num_workers = 1
        self.timeout = None
        self.bootstrap_from_timeout = True

        # optimizer
        self.opt_name = 'Adam'
        self.sgd_update_frequency = 1
        self.initialization = 'kaiming_uniform'
        self.gradient_clip = None
        self.clip_denom = None
        self.beta3 = None
        
        # normalization
        self.reg = None
        self.reg_weight = 1e-3
        
        """
        The following are algorithm/trick specific parameters
        """
        
        # n-step method
        self.rollout_length = 1
        
        # experience replay or planning
        self.memory_size = 50000
        self.batch_size = 32
        
        # target network
        self.use_target_network = False
        self.target_network_update_freq = 500
        self.target_network_mix = 0.001
        
        # initial exploration
        self.initial_exploration = False
        self.exploration_steps = 0
        
        # double q
        self.double_q = False

        if param_dict is not None:
            for key, value in param_dict.items():
                setattr(self, key, value)

    @property
    def eval_task(self):
        return self.__eval_task

    @eval_task.setter
    def eval_task(self, task):
        self.__eval_task = task
        self.state_dim = task.state_dim
        self.action_dim = task.action_dim
        self.task_name = task.env_id

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])

    # Helper functions
    def get_logdir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting))
        ensure_dir(d)
        return d

    def get_logdir_format(self):
        return os.path.join(self.data_root, self.exp_name,
                            "{}_run",
                            "{}_param_setting".format(self.param_setting))

    def get_tflogdir(self):
        return os.path.join(self.data_root, self.exp_name,
                            "{}_run".format(self.run),
                            "{}_param_setting".format(self.param_setting),
                            "tf_logs", datetime.now().strftime('%D-%T'))

    # Helper functions
    def get_modeldir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "model")
        ensure_dir(d)
        return d

    def log_config(self, logger):
        # Serializes the configure into a log file
        attrs = self.get_print_attrs()
        for param, value in sorted(attrs.items(), key=lambda x: x[0]):
            logger.info('{}: {}'.format(param, value))

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['state_normalizer', 'reward_normalizer', 'task_fn',
                  'logger', '_SlimConfig__eval_env', 'random_action_prob',
                  'optimizer_fn', 'network_fn', 'replay_fn', 'data_root']:
            del attrs[k]
        return attrs