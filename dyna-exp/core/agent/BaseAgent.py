#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np
from ..utils.misc import close_obj, mkdir
import torch.multiprocessing as mp
from collections import deque
import pickle


class BaseAgent:
	def __init__(self, config):
		self.config = config
		self.task_ind = 0
	
	def close(self):
		close_obj(self.task)
	
	def save(self, filename):
		torch.save(self.network.state_dict(), '%s.model' % (filename))
		with open('%s.stats' % (filename), 'wb') as f:
			pickle.dump(self.config.state_normalizer.state_dict(), f)
	
	def load(self, filename):
		state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
		self.network.load_state_dict(state_dict)
		with open('%s.stats' % (filename), 'rb') as f:
			self.config.state_normalizer.load_state_dict(pickle.load(f))
	
	def eval_step(self, state, reward, done, info, ep=None):
		raise NotImplementedError
	
	def eval_n_steps(self):
		task = self.config.eval_task
		n = self.config.eval_steps
		state = task.reset()
		reward = 0
		done = None
		info = None
		ret = 0
		for i in range(n):
			action = self.eval_step(state, reward, done, info)
			state, reward, done, info = task.step(action)
			ret += reward
		return ret * 1.0 / n
	
	def eval_episode(self, ep):
		task = self.config.eval_task
		state = task.reset()
		discounted_episodic_return = 0
		discounting = 1
		reward = 0
		done = True
		info = None
		state_list = [state]
		while True:
			action = self.eval_step(state, reward, done, info, ep)
			state, reward, done, info = task.step(action)
			state_list.append(state)
			discounted_episodic_return += discounting * reward
			discounting *= self.config.discount
			if done:
				break
		report = {}
		report.setdefault('discounted_episodic_return', []).append(discounted_episodic_return)
		report.setdefault('episodic_return', []).append(info['episodic_return'])
		report.setdefault('episodic_length', []).append(info['episodic_length'])
		arr = info['term']
		for i in range(arr.shape[0]):
			report.setdefault('term' + str(i), []).append(arr[i])
		return report
	
	def eval_episodes(self):
		reports = {}
		for ep in range(self.config.eval_episodes):
			report = self.eval_episode(ep)
			if report is not None:
				for key in report:
					reports.setdefault(key, []).extend(report[key])
		return reports
	
	def record_online_return(self, info, offset=0):
		if isinstance(info, dict):
			ret = info['episodic_return']
			if ret is not None:
				self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
				self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
		elif isinstance(info, tuple):
			for i, info_ in enumerate(info):
				self.record_online_return(info_, i)
		else:
			raise NotImplementedError
	
	def record_episode(self, dir, env):
		mkdir(dir)
		steps = 0
		state = env.reset()
		while True:
			self.record_obs(env, dir, steps)
			action = self.record_step(state)
			state, reward, done, info = env.step(action)
			ret = info[0]['episodic_return']
			steps += 1
			if ret is not None:
				break
	
	def record_step(self, state):
		raise NotImplementedError
	
	# For DMControl
	def record_obs(self, env, dir, steps):
		env = env.env.envs[0]
		obs = env.render(mode='rgb_array')


class BaseActor(mp.Process):
	STEP = 0
	RESET = 1
	EXIT = 2
	SPECS = 3
	NETWORK = 4
	CACHE = 5
	
	def __init__(self, config):
		mp.Process.__init__(self)
		self.config = config
		self.__pipe, self.__worker_pipe = mp.Pipe()
		
		self._state = None
		self._task = None
		self._network = None
		self._total_steps = 0
		self.__cache_len = 2
		
		# report
		self._save_network_flag = False
		self._log_flag = False
		self._eval_flag = False
		
		if not config.async_actor:
			self.start = lambda: None
			self.step = self._sample
			self.close = lambda: None
			self._set_up()
			self._task = config.task_fn(config.seed)
	
	def _sample(self):
		transitions = []
		for _ in range(self.config.sgd_update_frequency):
			entry = self._transition()
			if entry is not None:
				transitions.append(entry)
		return transitions
	
	def run(self):
		self._set_up()
		config = self.config
		self._task = config.task_fn(config.seed)
		
		cache = deque([], maxlen=2)
		while True:
			op, data = self.__worker_pipe.recv()
			if op == self.STEP:
				if not len(cache):
					cache.append(self._sample())
					cache.append(self._sample())
				self.__worker_pipe.send(cache.popleft())
				cache.append(self._sample())
			elif op == self.EXIT:
				self.__worker_pipe.close()
				return
			elif op == self.NETWORK:
				self._network = data
			else:
				raise NotImplementedError
	
	def _transition(self):
		raise NotImplementedError
	
	def _set_up(self):
		pass
	
	def step(self):
		self.__pipe.send([self.STEP, None])
		return self.__pipe.recv()
	
	def close(self):
		self.__pipe.send([self.EXIT, None])
		self.__pipe.close()
	
	def set_network(self, net):
		if not self.config.async_actor:
			self._network = net
		else:
			self.__pipe.send([self.NETWORK, net])
