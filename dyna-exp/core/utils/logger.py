#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import os
import torch
import logging
import datetime

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_logger(exp_name, sweep_id, log_level=0):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    assert exp_name is not None
    assert sweep_id is not None

    if not os.path.exists('./experiments/%s/outputs/log/' % exp_name):
        os.makedirs('./experiments/%s/outputs/log/' % exp_name)
    if os.path.exists('./experiments/%s/outputs/log/%s.txt' % (exp_name, str(sweep_id))):
        os.remove('./experiments/%s/outputs/log/%s.txt' % (exp_name, str(sweep_id)))
    fh = logging.FileHandler('./experiments/%s/outputs/log/%s.txt' % (exp_name, str(sweep_id)))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    
    return Logger(logger, './experiments/%s/outputs/tf_log/%s' % (exp_name, str(sweep_id)), log_level)


class Logger(object):
    def __init__(self, vanilla_logger, tf_log_dir, log_level=0):
        self.log_level = log_level
        # self.writer = SummaryWriter(tf_log_dir)
        if vanilla_logger is not None:
            self.info = vanilla_logger.info
            self.debug = vanilla_logger.debug
            self.warning = vanilla_logger.warning
        self.all_steps = {}
        self.tf_log_dir = tf_log_dir

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step