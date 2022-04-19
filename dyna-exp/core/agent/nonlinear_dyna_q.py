#######################################################################
# Copyright (C) 2022 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import torch.nn as nn
import numpy as np
from ..utils.torch_utils import tensor, range_tensor, to_np, set_optimizer
from ..utils.misc import close_obj
from .BaseAgent import BaseAgent
from ..component.replay import Replay
from ..network.network_utils import update_weights


class NonlinearDynaQ(BaseAgent):
	def __init__(self, config):
		BaseAgent.__init__(self, config)
		self.config = config
		self.task = config.task_fn(config.seed)

		self.learning_buffer = Replay(config.learning_buffer_size, config.learning_batch_size)
		if config.reward_emphasis:
			self.learning_buffer2 = Replay(config.learning_buffer_size, config.learning_batch_size)
		self.planning_buffer = Replay(config.planning_buffer_size, config.planning_batch_size)
		self.learning_batch_indices = range_tensor(config.learning_batch_size)
		self.planning_batch_indices = range_tensor(config.planning_batch_size)

		self.network = config.network
		if config.use_target_network:
			self.target_network = config.network_fn()
			self.target_network.load_state_dict(self.network.state_dict())
		else:
			self.target_network = self.network
		
		self.value_optimizer = set_optimizer(self.network.value_params, config)
		config.lr = config.model_lr
		self.model_optimizer = set_optimizer(self.network.model_params, config)

		self.total_steps = 0

		self.discounting = 1
		self.discounted_episodic_return = 0
		self.num_actions = config.action_space.n
		self.set_phase_and_eval(1)
		self.state = self.task.reset()
		self.network.train()
	
	def set_phase_and_eval(self, phase):
		print('set loca experiment phase to', phase)
		self.task.env.env.set_phase_and_eval(phase, False)
		self.config.eval_task.env.env.set_phase_and_eval(phase, True)

	def close(self):
		close_obj(self.replay)

	def eval_step(self, state, reward, done, info, ep):
		self.config.state_normalizer.set_read_only()
		state = self.config.state_normalizer(state)
		action = self._select_action(state, self.config.eval_epsilon)
		self.config.state_normalizer.unset_read_only()
		return action
	
	def _select_action(self, state, epsilon):
		self.network.eval()
		qs = self.network(np.expand_dims(state, axis=0), 'q')
		self.network.train()
		qs = to_np(qs).flatten()
		if np.random.rand() < epsilon:
			action = np.random.randint(0, self.num_actions)
		else:
			action = np.argmax(qs)
		return action
	
	def get_value(self, state):
		if len(state.shape) == 2:
			return to_np(self.network(state, 'q').max(1).values)
		elif len(state.shape) == 1:
			return to_np(self.network(np.expand_dims(state, axis=0), 'q').max().values)
		else:
			raise NotImplementedError
	
	def get_policy(self, state):
		if len(state.shape) == 2:
			a = np.argmax(to_np(self.network(state, 'q')), axis=1)
		elif len(state.shape) == 1:
			a = np.argmax(to_np(self.network(state, 'q')))
		else:
			raise NotImplementedError
		return a
	
	def get_reward(self, state):
		if len(state.shape) == 2:
			s,r,t = self.network(state, 'model')
		elif len(state.shape) == 1:
			s,r,t = self.network(np.expand_dims(state, axis=0), 'model')
		else:
			raise NotImplementedError
		return to_np(r)[:, :, 0].max(1)
	
	def get_term(self, state):
		if len(state.shape) == 2:
			s,r,t = self.network(state, 'model')
		elif len(state.shape) == 1:
			s,r,t = self.network(np.expand_dims(state, axis=0), 'model')
		else:
			raise NotImplementedError
		return to_np(t)[:, :, 0].max(1)
	
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

	def step(self):
		report = {}
		config = self.config
		
		action = self._select_action(self.state, config.epsilon)
		next_state, reward, done, info = self.task.step(action)

		self.learning_buffer.feed([
			self.state, action, config.reward_normalizer(reward), next_state,
			int(done) and not info['TimeLimit.truncated']
		])
		if config.reward_emphasis and reward != 0:
			self.learning_buffer2.feed([
				self.state, action, config.reward_normalizer(reward), next_state,
				int(done) and not info['TimeLimit.truncated']
			])
		self.planning_buffer.feed([self.state, np.random.randint(self.num_actions)])
		self.state = next_state
		self.discounted_episodic_return += self.discounting * reward
		self.discounting *= self.config.discount
		ret = info['episodic_return']
		if ret is not None:
			report.setdefault('episodic_return', []).append(info['episodic_return'])
			report.setdefault('episodic_length', []).append(info['episodic_length'])
			report.setdefault('discounted_episodic_return', []).append(self.discounted_episodic_return)
		if done:
			self.discounted_episodic_return = 0
			self.discounting = 1
		self.total_steps += 1
		
		to_print_dict = self.update()
		for key, value in to_print_dict.items():
			report.setdefault(key, []).append(value)

		if config.use_target_network and \
				self.total_steps / self.config.sgd_update_frequency % self.config.target_network_update_freq == 0:
			self.target_network.load_state_dict(self.network.state_dict())

		if self.total_steps % 1000000 == 0:
			self.network.eval()
			self.task.env.env.render_values(
				self.get_policy,
				'./experiments/%s/outputs/plots/' % self.config.exp_name,
				str(self.config.seed) + "_" + str(self.total_steps) + "_policy"
			)
			self.task.env.env.render_values(
				self.get_value,
				'./experiments/%s/outputs/plots/' % self.config.exp_name,
				str(self.config.seed) + "_" + str(self.total_steps) + "_value"
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
			self.network.train()
		return report
	
	def update(self):
		rtn_dict = {}
		for _ in range(self.config.num_learning_steps):
			batch_transitions = self.learning_buffer.sample()
			if batch_transitions is None:
				return {}
			states, actions, rewards, next_states, terminals = batch_transitions
			
			if self.config.reward_emphasis:
				batch_transitions2 = self.learning_buffer2.sample()
				if batch_transitions2 is None:
					return {}
				states2, actions2, rewards2, next_states2, terminals2 = batch_transitions2
				tmp = int(self.config.learning_batch_size/2)
				states = np.concatenate((states[:tmp], states2[tmp:]), axis=0)
				actions = np.concatenate((actions[:tmp], actions2[tmp:]), axis=0)
				rewards = np.concatenate((rewards[:tmp], rewards2[tmp:]), axis=0)
				next_states = np.concatenate((next_states[:tmp], next_states2[tmp:]), axis=0)
				terminals = np.concatenate((terminals[:tmp], terminals2[tmp:]), axis=0)
		
			# Model-learning
			s_loss, r_loss, t_loss, info = compute_model_loss(
				states, actions, rewards, next_states, terminals, self.network, self.learning_batch_indices
			)
			loss = s_loss + r_loss + t_loss
			update_weights(self.model_optimizer, loss, self.config.gradient_clip, self.network.model_params)
		rtn_dict.setdefault("s_loss", to_np(s_loss))
		rtn_dict.setdefault("r_loss", to_np(r_loss))
		rtn_dict.setdefault("t_loss", to_np(t_loss))
		
		# Planning
		for _ in range(self.config.num_planning_steps):
			states, actions = self.planning_buffer.sample()
			value_loss = compute_q_loss(
				states, self.network, self.target_network, self.config.discount, actions, self.planning_batch_indices
			)
			update_weights(self.value_optimizer, value_loss, self.config.gradient_clip, self.network.value_params)
		return rtn_dict


def compute_model_loss(
		states, actions, rewards, next_states, terminals, network,
		batch_indices
):
	next_states = tensor(next_states)
	terminals = tensor(terminals)
	rewards = tensor(rewards)
	actions = tensor(actions).long()
	states = tensor(states)

	pred_next_states, pred_rewards, pred_terminations = network(states, 'model')
	pred_next_states = pred_next_states[batch_indices, actions]
	pred_rewards = pred_rewards[batch_indices, actions]
	pred_terminations = pred_terminations[batch_indices, actions]
	state_loss = (next_states - pred_next_states).pow(2).mul(0.5).sum(1)
	state_loss = ((1 - terminals) * state_loss).mean()
	reward_loss = (rewards - pred_rewards[:, 0]).pow(2).mul(0.5).mean()
	criterion = nn.BCELoss()
	term_loss = criterion(pred_terminations[:, 0], terminals)
	info = {
		'pred_next_s': to_np(pred_next_states[0]),
		'next_s': to_np(next_states[0]),
		'pred_next_r': to_np(pred_rewards[0]),
		'next_r': to_np(rewards[0]),
		'pred_next_t': to_np(pred_terminations[0]),
		'next_t': to_np(terminals[0]),
		'r_loss_max': (rewards - pred_rewards).abs().max(),
		't_loss_max': (terminals - pred_terminations).abs().max()
	}
	return state_loss, reward_loss, term_loss, info


def compute_q_loss(states, network, target_network, discount, action, batch_indices):
	network.eval()
	next_state, reward, termination = network(states, 'model')
	network.train()
	reward = reward[:, :, 0][batch_indices, action]
	termination = termination[:, :, 0][batch_indices, action]
	next_state = next_state[batch_indices, action]
	
	r = reward
	t = termination
	s = next_state
	backup_values7 = r + discount * (1 - t) * target_network(s, 'q').max(1).values
	
	q_loss = (backup_values7.detach() - network(states, 'q')[batch_indices, action]).pow(2).mul(0.5).mean()
	return q_loss