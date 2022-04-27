#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import re
import gym
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from .wrappers import TimeLimit

from collections import defaultdict

from ..utils.misc import mkdir

try:
    import roboschool
except ImportError:
    pass


_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)


_game_envs['user_env'] = {
    'RiverSwim',
    'MountainCarLoCA',
    'FourRoomsLoCA',
    'ContMountainCar'
}


def get_env_type(env_id):

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type


# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(env_id, cfg, seed, timeout, episode_life=True):
    # TODO: add pygame in
    env_type = get_env_type(env_id)

    if env_type == 'dm':
        import dm_control2gym
        _, domain, task = env_id.split('-')
        env = dm_control2gym.make(domain_name=domain, task_name=task)
    elif env_type == 'user_env':
        from . import user_envs
        env_class = getattr(user_envs, env_id)
        env = env_class(cfg)
        if timeout is not None:
            env = TimeLimit(env, max_episode_steps=timeout)
    elif env_type == 'gym_pygame':
        from . import user_envs
        env_class = getattr(user_envs, env_id)
        env = env_class()
        if timeout is not None:
            env = TimeLimit(env, max_episode_steps=timeout)
    else:
        # other gym games
        env = gym.make(env_id)
        if timeout is not None:
            env = TimeLimit(env.unwrapped, max_episode_steps=timeout)

    env.seed(seed)

    return env


class Env(object):
    def __init__(self,
                 env_id,
                 cfg,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5)),
                 timeout=None
                 ):
        self.env_id = env_id
        self.timeout = timeout
        if log_dir is not None:
            mkdir(log_dir)
        env = make_env(env_id, cfg, seed, timeout, episode_life)
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.observation_space.shape))

        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'
        self.eps_return = 0
        self.eps_length = 0

    def step(self, action):
        if isinstance(self.action_space, Box):
            action = np.clip(action, self.action_space.low, self.action_space.high)

        obs, reward, done, info = self.env.step(action)
        self.eps_length += 1
        self.eps_return += reward
        
        if 'TimeLimit.truncated' not in info:
            info['TimeLimit.truncated'] = False
            
        if done:
            info['episodic_return'] = self.eps_return
            info['episodic_length'] = self.eps_length
            self.eps_length = 0
            obs = self.reset()
        else:
            info['episodic_return'] = None
            info['episodic_length'] = None
        return obs, reward, done, info

    def reset(self):
        self.eps_return = 0
        return self.env.reset()
    
    def set_eval(self):
        self.env.set_eval()

    def close(self):
        self.env.close()


if __name__ == '__main__':
    task = Env('Hopper-v2', None)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)
