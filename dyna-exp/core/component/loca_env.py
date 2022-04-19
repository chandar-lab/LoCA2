#######################################################################
# Copyright (C) 2022 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import gym


class LoCAEnv(gym.Env):
	"""
	This class is an abstract class that should be inherited by LoCA envs.
	The methods of this class should be implemented by the env designer.
	"""
	
	def set_phase_and_eval(self, phase, eval):
		"""
		LoCA is an experiment setting that consists 3 phases : pretraining 1, pretraining 2, and training.
		The second and the third phases share the same reward setting, which is different from it of the first phase.
		In the first phase and the third phase, the initial state is sampled uniformly randomly from the entire state
		space. In the second phase, the initial state is sampled from local regions where the rewards are changed.
		Right now, this function only needs to change rewards and the initial state distribution. For future extensions,
		the dynamics of the environment might also be changed.
		:param phase: '1', '2', '3' for pretraining 1, pretraining 2, and training respectively
		:return: None, if phase is not valid, assert an error.
		"""
		raise NotImplementedError