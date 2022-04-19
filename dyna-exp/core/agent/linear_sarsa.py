#######################################################################
# Copyright (C) 2022 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


from .BaseAgent import BaseAgent
import numpy as np


class SarsaLmbdaTileCoding(BaseAgent):
	def __init__(self, config):
		BaseAgent.__init__(self, config)
		self.config = config
		self.gamma = config.discount
		self.alpha = config.lr / config.num_tilings
		self.lAmbda = config.lAmbda
		self.epsilon = config.epsilon_init
		self.task = config.task_fn(config.seed)
		self.num_actions = self.task.action_space.n
		
		self.num_tilings = config.num_tilings
		self.num_tiles = config.num_tiles
		self.num_active_features = self.num_tilings
		self.num_total_state_features = self.num_tilings * np.prod(self.num_tiles)
		self.num_total_action_features = self.num_total_state_features * self.num_actions
		self.tiling_offsets = np.zeros((self.task.observation_space.shape[0], self.num_tilings))
		self._tile_coding_initialize()
		
		self.theta = np.zeros(self.num_total_action_features)
		self.e_trace = np.zeros(self.num_total_action_features)
		self.Qs_old = 0
		self.type = 2
		self.set_phase_and_eval(1)
		self.state = self.task.reset()
		state_features = np.zeros(self.num_active_features)
		self._get_state_features(self.state, state_features)
		self.action = self._select_action(state_features, self.epsilon)
		self.action_features = self._get_action_features(state_features, self.action)
		self.total_steps = 0
		self.episode_count = 0
		self.r_bar = 0
		self.discounted_episodic_return = 0
		self.discounting = 1
	
	def set_phase_and_eval(self, phase):
		if hasattr(self.task.env.env, "set_phase_and_eval"):
			print('set loca experiment phase to', phase)
			self.task.env.env.set_phase_and_eval(phase, False)
			self.config.eval_task.env.env.set_phase_and_eval(phase, True)
		
	def close(self):
		pass
	
	def eval_step(self, state, reward, done, info, ep):
		state_features = np.zeros(self.num_active_features)
		self._get_state_features(state, state_features)
		action = self._select_action(state_features, 0)
		return action
	
	def step(self):
		report = {}
		next_state, reward, done, info = self.task.step(self.action)
		next_state_features = np.zeros(self.num_active_features)
		self._get_state_features(next_state, next_state_features)
		next_action = self._select_action(next_state_features, self.epsilon)
		next_action_features = self._get_action_features(next_state_features, next_action)
		td_error = self._update_theta(self.action_features, reward, next_action_features, done)
		self.action_features = next_action_features
		self.action = next_action
		self.total_steps += 1
		if done:
			self.episode_count += 1
			self._decay_epsilon()
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
		self.discounted_episodic_return += self.discounting * reward
		self.discounting *= self.config.discount
		report.setdefault('td_error', []).append(td_error)
		report.setdefault('q_values', []).append(self.theta[self.action_features])
		report.setdefault('r_bar', []).append(self.r_bar)
		report.setdefault('reward', []).append(reward)
		if info['episodic_return'] is not None:
			report.setdefault('episodic_return', []).append(info['episodic_return'])
			report.setdefault('episodic_length', []).append(info['episodic_length'])
			report.setdefault('discounted_episodic_return', []).append(self.discounted_episodic_return)
		if done:
			self.discounted_episodic_return = 0
			self.discounting = 1
		return report
	
	def _decay_epsilon(self):
		epsilon = self.config.epsilon_init * self.config.ep_decay_rate ** (self.episode_count / self.config.epsilon_decay_steps)
		self.epsilon = max(epsilon, self.config.min_epsilon)
	
	def _initialize_trace(self):
		for i in range(self.num_total_action_features):
			self.e_trace[i] = 0
		self.Qs_old = 0
	
	def _update_theta(self, update_features, reward, update_features2, done):
		
		Qs = sum(self.theta[update_features])
		assert np.all(update_features2 >= 0) or np.all(update_features2 == -1)
		if done is True:
			Qs2 = 0
		else:
			Qs2 = sum(self.theta[update_features2])
		
		delta = reward + self.gamma * Qs2 - Qs
		
		if self.type == 2:
			delta += Qs - self.Qs_old
		
		# update traces
		if self.type == 0:  # accumulating traces
			self.e_trace[update_features] += self.alpha
		elif self.type == 1:  # replacing traces
			self.e_trace[update_features] = self.alpha
		elif self.type == 2:  # dutch traces
			e_phi = sum(self.e_trace[update_features])
			self.e_trace[update_features] += self.alpha * (1 - e_phi)
		else:
			assert False
		
		delta = delta - self.r_bar
		self.theta += self.e_trace * delta
		
		if self.type == 2:
			self.theta[update_features] -= self.alpha * (Qs - self.Qs_old)
			self.Qs_old = Qs2
		
		if done is True:
			self._initialize_trace()
		else:
			self.e_trace *= self.gamma * self.lAmbda
		
		return delta
				
	def _select_action(self, state_features, epsilon):
		if epsilon >= 1.0:
			return np.random.randint(0, self.num_actions - 1)
		
		# determine Q-values
		Q = [0.0] * self.num_actions
		for a in range(self.num_actions):
			for j in range(self.num_active_features):
				f = int(state_features[j]) + a * self.num_total_state_features
				Q[a] += self.theta[f]
		
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
		
		return action
	
	def _get_action_features(self, state_features, action):
		action_features = np.zeros(self.num_active_features, dtype=np.int)
		for i in range(self.num_active_features):
			action_features[i] = int(state_features[i] + action * self.num_total_state_features)
		return action_features
	
	def _get_state_features(self, state, state_features):
		self._get_active_state_features(state, state_features)
	
	def _get_active_state_features(self, state, state_features):
		dim = state.shape[0]
		low = self.task.observation_space.low
		high = self.task.observation_space.high
		tile_sizes = [(high[i] - low[i]) / float(self.num_tiles[i] - 1) for i in range(dim)]

		for t in range(self.num_active_features):
			tmp = state + self.tiling_offsets[:, t]
			ftmp = np.minimum(np.divide(tmp - low, tile_sizes).astype(int), np.array(self.num_tiles))
			ft = ftmp[0] + t * np.prod(self.num_tiles)
			for i in range(1, dim):
				ft += ftmp[i] * self.num_tiles[i-1]
			assert (0 <= ft < self.num_total_state_features)
			state_features[t] = ft
	
	def _tile_coding_initialize(self):
		low = self.task.observation_space.low
		high = self.task.observation_space.high
		tile_sizes = [(high[i] - low[i]) / float(self.num_tiles[i] - 1) for i in range(low.shape[0])]
		
		for t in range(self.num_tilings):
			for i in range(len(tile_sizes)):
				self.tiling_offsets[i, t] = np.random.uniform(0, tile_sizes[i])
	
	def get_value(self, state):
		value_list = []
		for i in range(state.shape[0]):
			state_features = np.zeros(self.num_active_features)
			self._get_state_features(state[i], state_features)
			# determine Q-values
			Q = [0.0] * self.num_actions
			for a in range(self.num_actions):
				for j in range(self.num_active_features):
					f = int(state_features[j]) + a * self.num_total_state_features
					Q[a] += self.theta[f]
			value_list.append(max(Q))
		return np.array(value_list)
	
	def get_policy(self, state):
		value_list = []
		for i in range(state.shape[0]):
			state_features = np.zeros(self.num_active_features)
			self._get_state_features(state[i], state_features)
			# determine Q-values
			Q = [0.0] * self.num_actions
			for a in range(self.num_actions):
				for j in range(self.num_active_features):
					f = int(state_features[j]) + a * self.num_total_state_features
					Q[a] += self.theta[f]
			value_list.append(np.argmax(Q))
		return np.array(value_list)