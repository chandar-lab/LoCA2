#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import os
from pathlib import Path
import copy
import inspect


def get_args(func):
	signature = inspect.signature(func)
	return [k for k, v in signature.parameters.items()]


def get_default_args(func):
	signature = inspect.signature(func)
	return {
		k: v.default
		for k, v in signature.parameters.items()
		if v.default is not inspect.Parameter.empty
	}


def shallow_copy(obj):
	return copy.copy(obj)


def deep_copy(obj):
	return copy.deepcopy(obj)


def mkdir(path):
	Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
	if hasattr(obj, 'close'):
		obj.close()


def random_sample(indices, batch_size):
	indices = np.asarray(np.random.permutation(indices))
	batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
	for batch in batches:
		yield batch
	r = len(indices) % batch_size
	if r:
		yield indices[-r:]


def generate_tag(params):
	if 'tag' in params.keys():
		return
	game = params['game']
	params.setdefault('run', 0)
	run = params['run']
	del params['game']
	del params['run']
	str = ['%s_%s' % (k, v) for k, v in sorted(params.items())]
	tag = '%s-%s-run-%d' % (game, '-'.join(str), run)
	params['tag'] = tag
	params['game'] = game
	params['run'] = run


def translate(pattern):
	groups = pattern.split('.')
	pattern = ('\.').join(groups)
	return pattern


def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def ensure_dir(d):
	if not os.path.exists(d):
		os.makedirs(d)


class RunningMeanStd(object):
	# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
	def __init__(self, epsilon=1e-4, shape=()):
		self.mean = np.zeros(shape, 'float64')
		self.var = np.ones(shape, 'float64')
		self.count = epsilon
	
	def update(self, x):
		batch_mean = np.mean(x, axis=0)
		batch_var = np.var(x, axis=0)
		batch_count = x.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)
	
	def update_from_moments(self, batch_mean, batch_var, batch_count):
		self.mean, self.var, self.count = update_mean_var_count_from_moments(
			self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
	delta = batch_mean - mean
	tot_count = count + batch_count
	
	new_mean = mean + delta * batch_count / tot_count
	m_a = var * count
	m_b = batch_var * batch_count
	M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
	new_var = M2 / tot_count
	new_count = tot_count
	
	return new_mean, new_var, new_count