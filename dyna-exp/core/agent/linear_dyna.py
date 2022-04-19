#######################################################################
# Copyright (C) 2022 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from ..utils.misc import close_obj
from .BaseAgent import BaseAgent
from ..component.replay import Replay
from ..utils.tiles3 import tiles, IHT


class LinearDyna(BaseAgent):
	def __init__(self, config):
		BaseAgent.__init__(self, config)
		self.config = config
		self.replay = Replay(config.memory_size, config.batch_size)
		self.total_steps = 0
		self.discounted_episodic_return = 0
		self.discounting = 1
		self.num_planning_steps = self.config.num_planning_steps
		self.task = config.task_fn(config.seed)
		self.state_dim = self.task.observation_space.shape[0]
		self.num_actions = self.task.action_space.n
		self.low = self.task.observation_space.low
		self.high = self.task.observation_space.high
		self.num_tilings = self.config.num_tilings
		self.num_tiles = self.config.num_tiles
		self.lr = self.config.lr / self.num_tilings
		self.model_lr = self.config.model_lr / self.num_tilings
		self._tile_coding_initialize()
		self.F = np.zeros((self.num_actions, self.feature_dim, self.feature_dim))
		self.b = np.zeros((self.num_actions, self.feature_dim))
		self.theta = np.zeros(self.feature_dim)
		self.set_phase_and_eval(1)
		self.state = self.task.reset()
		self.phi = self.get_features(self.state)
		if self.config.search_control == "experience replay":
			self.replay = Replay(self.config.buffer_size, 1)
	
	def get_features(self, state):

		features = tiles(self.iht, self.num_tilings, self.scale * state, [])
		return features
	
	def print_features(self):
		arr = np.zeros(self.feature_dim)
		for i in range(self.task.env.env.num_states):
			state = np.array(self.task.env.env.tocell[i])
			arr[self.get_features(state)[0]] += 1
		print(arr)
	
	def set_phase_and_eval(self, phase):
		print('set loca experiment phase to', phase)
		self.task.env.env.set_phase_and_eval(phase, False)
		self.config.eval_task.env.env.set_phase_and_eval(phase, True)
	
	def close(self):
		close_obj(self.replay)
	
	def eval_step(self, state, reward, done, info, ep):
		state_features = self.get_features(state)
		action, _ = self._select_action(state_features, 0.0)
		return action
	
	def step(self):
		report = {'episodic_return': []}
		a, is_greedy = self._select_action(self.phi, self.config.epsilon)
		next_state, reward, done, info = self.task.step(a)
		
		self.discounted_episodic_return += self.discounting * reward
		self.discounting *= self.config.discount
		ret = info['episodic_return']
		if ret is not None:
			report['episodic_return'].append(ret)
			report.setdefault('episodic_length', []).append(info['episodic_length'])
			report.setdefault('discounted_episodic_return', []).append(self.discounted_episodic_return)
		if done:
			self.discounted_episodic_return = 0
			self.discounting = 1
			
		phi_prime = self.get_features(next_state)
		if is_greedy:
			self._update_theta_using_real_transition(self.phi, reward, phi_prime, done)
		self._update_F(a, self.phi, phi_prime, done)
		self._update_b(a, self.phi, reward)

		if self.total_steps % self.config.model_eval_interval == 0:
			d_error_list = []
			r_error_list = []
			for _ in range(self.config.model_eval_samples):
				s = self.task.env.env.get_a_random_state()
				a = np.random.randint(self.num_actions)
				features = self.get_features(s)
				s_prime, r, done = self.task.env.env.true_model_step(s, a)
				next_features = self.get_features(s_prime)
				d_error = self._compute_dynamics_model_error(a, features, next_features, done)
				r_error = self._compute_reward_model_error(a, features, r)
				d_error_list.append(d_error)
				r_error_list.append(r_error)
			report.setdefault('d_error', []).append(np.mean(d_error_list))
			report.setdefault('r_error', []).append(np.mean(r_error_list))
		if self.config.search_control == "experience replay":
			self.replay.feed(self.phi)
				
		self.planning()
		
		self.phi = phi_prime
		self.state = next_state
		self.total_steps += 1
		
		if self.total_steps % 1000000 == 0:
			self.task.env.env.render_values(
				self.get_value,
				'./experiments/%s/outputs/plots/' % self.config.exp_name,
				str(self.config.seed) + "_" + str(self.total_steps) + "_value"
			)
			self.task.env.env.render_values(
				self.get_policy,
				'./experiments/%s/outputs/plots/' % self.config.exp_name,
				str(self.config.seed) + "_" + str(self.total_steps) + "_policy"
			)
			self.task.env.env.render_values(
				self.get_reward,
				'./experiments/%s/outputs/plots/' % self.config.exp_name,
				str(self.config.seed) + "_" + str(self.total_steps) + "_reward"
			)
			self.task.env.env.render_values(
				self.get_term,
				'./experiments/%s/outputs/plots/' % self.config.exp_name,
				str(self.config.seed) + "_" + str(self.total_steps) + "_term"
			)

		return report
	
	def get_value(self, state):
		value_list = []
		for i in range(state.shape[0]):
			value_list.append(sum(self.theta[self.get_features(state[i])]))
		return np.array(value_list)
	
	def get_policy(self, state):
		action_list = []
		for i in range(state.shape[0]):
			action_list.append(self._select_action(self.get_features(state[i]), 0.0)[0])
		return np.array(action_list)
	
	def get_reward(self, state):
		reward_list = []
		for i in range(state.shape[0]):
			reward_list.append(np.max(np.sum(self.b[:, self.get_features(state[i])], axis=1)))
		return np.array(reward_list)
	
	def get_term(self, state):
		term_list = []
		for i in range(state.shape[0]):
			term_list.append(np.max(np.dot(np.sum(self.F[:, :, self.get_features(state[i])], axis=2), self.theta)))
		return np.array(term_list)
	
	def get_true_reward(self, state):
		rtn = []
		for i in range(state.shape[0]):
			_, r0, _ = self.task.env.true_model_step(state[i], 0)
			_, r1, _ = self.task.env.true_model_step(state[i], 1)
			_, r2, _ = self.task.env.true_model_step(state[i], 2)
			rtn.append(max(r0, r1, r2))
		return np.array(rtn)
	
	def get_true_term(self, state):
		rtn = []
		for i in range(state.shape[0]):
			_, _, t0 = self.task.env.true_model_step(state[i], 0)
			_, _, t1 = self.task.env.true_model_step(state[i], 1)
			_, _, t2 = self.task.env.true_model_step(state[i], 2)
			rtn.append(max(t0, t1, t2))
		return np.array(rtn)
	
	def _search_control(self):
		if self.config.search_control == "random feature":
			# Randomly generate one feature.
			features = [np.random.randint(self.feature_dim)]
		elif self.config.search_control == "experience replay":
			# Uniformly sample a state from a replay buffer, then generate the state's features.
			sampled_indice = np.random.randint(0, len(self.replay.data))
			features = self.replay.data[sampled_indice]
		else:
			raise NotImplementedError
		return features
	
	def planning(self):
		for i in range(self.num_planning_steps):
			features = self._search_control()
			self._update_theta_using_model(features)
			
	def _select_action(self, state_features, epsilon):
		Q = np.sum(self.b[:, state_features], axis=1) + np.dot(np.sum(self.F[:, :, state_features], axis=2), self.theta)
		# determine Qmax & num_max of the array Q[s]
		Qmax = Q[0]
		num_max = 1
		for i in range(1, self.num_actions):
			if Q[i] > Qmax:
				Qmax = Q[i]
				num_max = 1
			elif Q[i] == Qmax:
				num_max += 1
		
		# simultaneously compute selection probability for each action and select action
		rnd = np.random.random()
		cumulative_prob = 0.0
		action = self.num_actions - 1
		for a in range(self.num_actions - 1):
			prob = epsilon / float(self.num_actions)
			if Q[a] == Qmax:
				prob += (1 - epsilon) / float(num_max)
			cumulative_prob += prob
			
			if rnd < cumulative_prob:
				action = a
				break
		
		return action, Qmax == Q[action]
	
	def _update_theta_using_real_transition(self, update_features, reward, next_state_features, done):
		
		Vs = sum(self.theta[update_features])
		if done:
			Vs2 = 0
		else:
			Vs2 = sum(self.theta[next_state_features])
		
		delta = reward + self.config.discount * Vs2 - Vs
		
		self.theta[update_features] += self.lr * delta
		return delta
	
	def _update_theta_using_model(self, update_features, a=None):
		if a is None:
			delta = np.max(
				np.sum(self.b[:, update_features], axis=1) +
				self.config.discount * np.dot(np.sum(self.F[:, :, update_features], axis=2), self.theta)
			) - sum(self.theta[update_features])
		else:
			delta = np.sum(self.b[a, update_features]) + self.config.discount * np.dot(np.sum(self.F[a][:, update_features], axis=1), self.theta) - sum(self.theta[update_features])
		self.theta[update_features] += self.lr * delta
		return delta
	
	def _update_F(self, action, update_features, next_state_features, done):
		if done:
			next_state_features = []
		next_state_feature_vector = np.zeros(self.feature_dim)
		next_state_feature_vector[next_state_features] = 1
		tmp = next_state_feature_vector - np.sum(self.F[action][:, update_features], axis=1)
		for idx in update_features:
			self.F[action, :, idx] += self.model_lr * tmp
	
	def _update_b(self, action, update_features, reward):
		
		delta = reward - sum(self.b[action][update_features])
		
		self.b[action][update_features] += self.model_lr * delta
	
	def _compute_dynamics_model_error(self, action, update_features, next_state_features, done):
		if done:
			next_state_features = []
		next_state_feature_vector = np.zeros(self.feature_dim)
		next_state_feature_vector[next_state_features] = 1
		dynamics_model_error = np.sum(np.square(next_state_feature_vector - np.sum(self.F[action][:, update_features], axis=1)))
		return dynamics_model_error
	
	def _compute_reward_model_error(self, action, update_features, reward):
		dynamics_model_error = np.square(reward - sum(self.b[action][update_features]))
		return dynamics_model_error

	def _tile_coding_initialize(self):
		self.num_total_state_features = self.num_tilings * np.prod(self.num_tiles)
		self.feature_dim = self.num_total_state_features
		self.iht = IHT(self.feature_dim)
		self.scale = [float(self.num_tiles[i] - 1) / (self.high[i] - self.low[i]) for i in range(self.state_dim)]
	
	def render_policy(self):
		a = np.array([self._select_action(
			self.get_features(np.array(self.task.env.env.tocell[i])), 0) for i in range(self.task.env.env.num_states)])
		self.task.env.env.render_policy(a)
