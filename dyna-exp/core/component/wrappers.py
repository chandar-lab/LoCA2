#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import gym

"""
OpenAI Baselines's wrapper
"""


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        info['TimeLimit.truncated'] = False
        if self._elapsed_steps >= self._max_episode_steps and done is False:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def set_eval(self):
        if hasattr(self.env, 'set_eval'):
            self.env.set_eval()
        else:
            print("wrappers: set_eval method not found in the env")
    
    def close(self):
        self.env.close()