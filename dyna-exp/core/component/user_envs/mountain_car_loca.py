#######################################################################
# Copyright (C) 2022 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import numpy as np
from gym import spaces
from gym.utils import seeding
from ..loca_env import LoCAEnv
import os


class MountainCarLoCA(LoCAEnv):
	"""
	Description:
		For any given state the agent may choose to accelerate to the left, right or cease
		any acceleration.
	Source:
		A variant of OpenAI gym's MountainCar domain
	Observation:
		Type: Box(2)
		Num    Observation               Min            Max
		0      Car Position              -1.2           0.5
		1      Car Velocity              -0.07          0.07
	Actions:
		Type: Discrete(3)
		Num    Action
		0      Accelerate to the Left
		1      Don't accelerate
		2      Accelerate to the Right
		Note: This does not affect the amount of velocity affected by the
		gravitational pull acting on the car.
	Reward:
		phase1:
			Reward of 4 is awarded if the agent reached the flag (position = 0.5)
			on top of the mountain.
			Reward of 2 is awarded if the agent reached the bottom and the velocity is 0.
			Reward of 0 is awarded otherwise.
		phase2&3:
			Reward of 2 is awarded if the agent reached the bottom and the velocity is 0.
			Reward of 0 is awarded otherwise.
	Starting State:
		phase1:
			The position and the velocity of the car are assigned uniform random values in
			[-1.2, 0.5], [-0.07, 0.07] with 0.5 probability and in [-1, 0], [-0.03, 0.03] with 0.5 probability.
		phase2:
			The position and the velocity of the car are assigned uniform random values in
			[0.4, 0.5], [0, 0.07].
		phase3:
			Same with phase1.
	Episode Termination:
		The car position is 0.5.
		The car position is around -0.5234 and the velocity is around 0.
	"""
	
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 30
	}
	
	def __init__(self, cfg):
		self.x_range = [-1.2, 0.5]
		self.v_range = [-0.07, 0.07]
		self.goal_position = 0.5
		self.force = cfg.force
		self.gravity = 0.0025
		self.low = np.array([self.x_range[0], self.v_range[0]], dtype=np.float64)
		self.high = np.array([self.x_range[1], self.v_range[1]], dtype=np.float64)
		self.init_state_phase1 = [[[-1.2, 0.5], [-0.07, 0.07]]]
		self.init_state_phase2 = [[[0.4, 0.5], [0, 0.07]]]
		self.init_state_phase3 = [[[-1.2, 0.5], [-0.07, 0.07]]]
		self.init_state_eval = [[[-0.2, -0.1], [-0.01, 0.01]]]
		self.terminal2_radius = cfg.terminal2_radius
		self.action_stochasticity = cfg.action_stochasticity
		self.action_space = spaces.Discrete(3)
		self.observation_space = spaces.Box(self.low, self.high, dtype=np.float64)
		self.init_state = None
		self.reward_terminal1 = None
		self.reward_terminal2 = None
		self.set_phase_and_eval(1, False)  # default
		self.seed()
	
	def render_values(self, get_value, save_folder, title, network=None):
		num = 100
		state_arr = np.zeros((num + 1, num + 1, 2))
		step = (self.high - self.low) / num
		for i in range(num+1):
			for j in range(num+1):
				state_arr[i, j, :] = self.low + np.array([i * step[0], j * step[1]])
		if network is None:
			values = get_value(state_arr.reshape((num + 1) * (num + 1), 2))
		else:
			values = get_value(state_arr.reshape((num + 1) * (num + 1), 2), network)
		plot_arr = values.reshape(num + 1, num + 1).T
		plot_arr = np.array([plot_arr[num - i] for i in range(num + 1)])
		if not os.path.exists(save_folder):
			os.makedirs(save_folder)
		np.save(save_folder + title, plot_arr)
	
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
		
	def set_phase_and_eval(self, phase, eval):
		if eval:
			self.init_state = self.init_state_eval
			if phase == 1:
				self.reward_terminal1 = 4
				self.reward_terminal2 = 2
			elif phase == 2:
				self.reward_terminal1 = 1
				self.reward_terminal2 = 2
			elif phase == 3:
				self.reward_terminal1 = 1
				self.reward_terminal2 = 2
			else:
				assert False, 'incorrect identifier'
		else:
			if phase == 1:
				self.init_state = self.init_state_phase1
				self.reward_terminal1 = 4
				self.reward_terminal2 = 2
			elif phase == 2:
				self.init_state = self.init_state_phase2
				self.reward_terminal1 = 1
				self.reward_terminal2 = 2
			elif phase == 3:
				self.init_state = self.init_state_phase3
				self.reward_terminal1 = 1
				self.reward_terminal2 = 2
			else:
				assert False, 'incorrect identifier'
	
	def step(self, action):
		
		x, v = self.state
		if np.random.random() < self.action_stochasticity:
			action = np.random.randint(3)
		if x >= 0.4 and v >= 0:
			action = 2
		next_v = v + self.force * (action - 1) - self.gravity * math.cos(3 * x)
		next_v = np.clip(next_v, self.v_range[0], self.v_range[1])
		next_x = x + next_v
		next_x = np.clip(next_x, self.x_range[0], self.x_range[1])
		if next_x == self.x_range[0] and next_v < 0:
			next_v = 0
		info = {}
		if next_x >= self.goal_position and next_v >= 0:
			done = True
			info['term'] = np.array([0, 1, 0])
			reward = self.reward_terminal1
		elif ((next_x + 0.52) ** 2 + 100 * next_v ** 2) <= self.terminal2_radius**2:
			done = True
			info['term'] = np.array([0, 0, 1])
			reward = self.reward_terminal2
		else:
			done = False
			info['term'] = np.array([1, 0, 0])
			reward = 0
		self.state = (next_x, next_v)
		return np.array(self.state), reward, done, info
	
	def reset(self):
		reset = False
		x, v = 0, 0
		p = np.random.uniform(0, 1)
		while not reset:
			if len(self.init_state) == 1:
				x = np.random.uniform(self.init_state[0][0][0], self.init_state[0][0][1])
				v = np.random.uniform(self.init_state[0][1][0], self.init_state[0][1][1])
			elif len(self.init_state) == 2:  # a mix distribution for initial states
				if p < 0.5:
					x = np.random.uniform(self.init_state[0][0][0], self.init_state[0][0][1])
					v = np.random.uniform(self.init_state[0][1][0], self.init_state[0][1][1])
				else:
					x = np.random.uniform(self.init_state[1][0][0], self.init_state[1][0][1])
					v = np.random.uniform(self.init_state[1][1][0], self.init_state[1][1][1])
			
			if ((x + 0.5234) ** 2 + 100 * v ** 2) >= (self.terminal2_radius)**2:
				reset = True
		self.state = (x, v)
		return np.array(self.state)
	
	def _height(self, xs):
		return np.sin(3 * xs) * .45 + .55
		
	def close(self):
		pass
			
	def get_a_random_state(self):
		x = np.random.uniform(-1.2, 0.5)
		v = np.random.uniform(-0.07, 0.07)
		return np.array([x, v])
	
	def true_model_step(self, s, a):
		x, v = s
		if x >= 0.4 and v >= 0:
			a = 2
		if np.random.random() < self.action_stochasticity:
			a = np.random.randint(3)
		next_v = v + self.force * (a - 1) - self.gravity * math.cos(3 * x)
		next_v = np.clip(next_v, self.v_range[0], self.v_range[1])
		next_x = x + next_v
		next_x = np.clip(next_x, self.x_range[0], self.x_range[1])
		if next_x == self.x_range[0] and next_v < 0:
			next_v = 0
		if next_x >= self.goal_position and next_v >= 0:
			done = True
			reward = self.reward_terminal1
		elif ((next_x + 0.52) ** 2 + 100 * next_v ** 2) <= self.terminal2_radius**2:
			done = True
			reward = self.reward_terminal2
		else:
			done = False
			reward = 0
		return np.array((next_x, next_v)), reward, done