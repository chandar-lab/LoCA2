################################################
#  Author: Harm van Seijen
#  Copyright 2022 Microsoft
################################################



import numpy as np


class ModelBasedAgent(object):

    def __init__(self, settings, domain):

        self.domain = domain
        self.alpha = settings['alpha']
        self.epsilon_default = settings['epsilon']
        self.epsilon = self.epsilon_default
        self.r_init = settings['q_init']
        self.eval_episodes = settings['eval_episodes']
        self.eval_epsilon = settings['eval_epsilon']
        self.eval_max_steps = settings['eval_max_steps']
        self.max_episode_length = settings['max_episode_length']
        if settings['method'] == 'mb_1step_random':
            self.update_v_current_state = False
            self.nstep = 1
        elif settings['method'] == 'mb_1step_current':
            self.update_v_current_state = True
            self.nstep = 1
        elif settings['method'] == 'mb_2_step_random':
            self.update_v_current_state = False
            self.nstep = 2
        else:
            assert False, 'HvS: Invalid method id.'
        self.num_states = domain.get_num_states()
        self.num_actions = domain.get_num_actions()
        self.gamma = domain.get_gamma()
        self.trans_model = None
        self.trans_model_phase1 = None
        self.trans_model_phase2 = None
        self.reward_model = None
        self.reward_model_phase1 = None
        self.reward_model_phase2 = None
        self.v = None
        self.v_phase1 = None
        self.v_phase2 = None

        # replay buffer is used when nstep > 1
        self.buffer_num_samples = None
        self.buffer_states = None
        self.buffer_actions = None
        self.buffer_rewards = None

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def reset_epsilon(self):
        self.epsilon = self.epsilon_default

    def _initialize(self):
        self.trans_model = np.zeros([self.num_states, self.num_actions, self.num_states])
        self.reward_model = np.ones([self.num_states, self.num_actions]) * self.r_init
        self.v = np.ones([self.num_states]) * self.r_init

    def run_single_task(self, steps, eval_window_size, single_task_id, rnd_explore_steps=0):
        print("Singe-task run started.")
        self.domain.set_task(single_task_id)  # 0 : A ,   1: B
        self.domain.set_phase(0)  # determines initial state distribution
        self._initialize()

        perf = self.run(steps, eval_window_size, rnd_explore_steps)

        print("Single-task avg return: ", np.mean(perf), " final return: ", perf[-1])
        return perf

    def run_phase1(self, steps, eval_window_size, rnd_explore_steps=0):
        print("Phase 1 started.")
        self.domain.set_task(0)  # set domain to task A
        self.domain.set_phase(1)
        self._initialize()

        perf = self.run(steps, eval_window_size, rnd_explore_steps)  # HvS: No longer fixed behavior during transition

        self.trans_model_phase1 = np.copy(self.trans_model)
        self.reward_model_phase1 = np.copy(self.reward_model)
        self.v_phase1 = np.copy(self.v)

        print("Phase 1: avg return: ", np.mean(perf), " final return: ", perf[-1])
        return perf

    def run_phase2(self, steps, eval_window_size):
        print("Phase 2 started.")
        self.domain.set_task(1)  # set domain to task B
        self.domain.set_phase(2)

        assert self.trans_model_phase1 is not None
        self.trans_model = np.copy(self.trans_model_phase1)
        self.reward_model = np.copy(self.reward_model_phase1)
        self.v = np.copy(self.v_phase1)

        perf = self.run(steps, eval_window_size)
        self.trans_model_phase2 = np.copy(self.trans_model)
        self.reward_model_phase2 = np.copy(self.reward_model)
        self.v_phase2 = np.copy(self.v)

        print("Phase 2: avg return: ", np.mean(perf), " final return: ", perf[-1])
        return perf

    def run_phase3(self, steps, eval_window_size):
        print("Phase 3 started.")
        self.domain.set_task(1)  # set domain to task B
        self.domain.set_phase(3)

        assert self.trans_model_phase2 is not None
        self.trans_model = np.copy(self.trans_model_phase2)
        self.reward_model = np.copy(self.reward_model_phase2)
        self.v = np.copy(self.v_phase2)

        perf = self.run(steps, eval_window_size)

        print("Phase 3: avg return: ", np.mean(perf), " final return: ", perf[-1])
        return perf

    def run(self, steps, window_size, rnd_explore_steps=0):
        # perform a run over certain number of time steps

        num_datapoints = steps // window_size
        data_point = 0
        self.epsilon = 1.0

        performance = np.zeros([num_datapoints])
        performance[data_point] = self.eval_policy()
        data_point += 1

        time = 0

        state = self.domain.get_initial_state()
        for i in range(steps):

            if i >= rnd_explore_steps:
                self.epsilon = self.epsilon_default

            # update state
            if self.update_v_current_state:
                update_state = state
            else:
                update_state = np.random.randint(self.num_states)
            self._update_state_value(update_state)

            action = self.select_action(state)
            next_state, reward = self.domain.take_action(state, action)
            time += 1

            if self.nstep == 1:
                self.update_model(state, action, reward, next_state)
            else:
                self.update_nstep_model(state, action, reward, next_state, time)

            if next_state == -1 or time == self.max_episode_length:
                time = 0
                state = self.domain.get_initial_state()
            else:
                state = next_state

            if ((i + 1) % window_size == 0) & (data_point < num_datapoints):
                performance[data_point] = self.eval_policy()
                data_point += 1

        return performance

    def update_model(self, state, action, reward, next_state):
        assert (state >= 0)
        next_state_vector = np.zeros([self.num_states])
        if next_state != -1:
            next_state_vector[next_state] = 1

        self.trans_model[state][action] += self.alpha * (next_state_vector - self.trans_model[state][action])
        self.reward_model[state][action] += self.alpha * (reward - self.reward_model[state][action])

    def update_nstep_model(self, state, action, reward, next_state, time):
        if time == 1:
            self.reset_experience_buffer()
        self.update_experience_buffer(state, action, reward)
        if next_state == -1:
            # perform multiple update
            delta_t = min(self.nstep, time)
            for t in range(time - delta_t, time):
                self._single_update_nstep_model(t, time, next_state)
        elif time >= self.nstep:
            self._single_update_nstep_model(time - self.nstep, time, next_state)
        else:
            return  # no updates

    def _single_update_nstep_model(self, t_update, t_current, next_state):
        state = self.buffer_states[t_update]
        action = self.buffer_actions[t_update]

        # state-model update:
        future_state_vector = np.zeros([self.num_states])
        if next_state != -1:
            future_state_vector[next_state] = 1
        self.trans_model[state][action] += self.alpha * (future_state_vector - self.trans_model[state][action])

        # reward-model update:
        nstep_return = 0
        discount = 1
        for i in range(t_update, t_current):
            reward = self.buffer_rewards[i]
            nstep_return += reward * discount
            discount *= self.gamma
        self.reward_model[state][action] += self.alpha * (nstep_return - self.reward_model[state][action])

    def reset_experience_buffer(self):
        self.buffer_num_samples = 0
        self.buffer_states = np.zeros(self.max_episode_length, dtype=np.int32)
        self.buffer_actions = np.zeros(self.max_episode_length, dtype=np.int32)
        self.buffer_rewards = np.zeros(self.max_episode_length)

    def update_experience_buffer(self, state, action, reward):
        index = self.buffer_num_samples
        self.buffer_states[index] = state
        self.buffer_actions[index] = action
        self.buffer_rewards[index] = reward
        self.buffer_num_samples += 1

    def _get_q(self):
        q = np.zeros([self.num_states, self.num_actions])
        for s in range(self.num_states):
            q[s] = self.compute_q_onestep_rollout(s)
        return q

    def _update_state_value(self, state):
        q = self.compute_q_onestep_rollout(state)
        self.v[state] = np.max(q)

    def compute_q_onestep_rollout(self, state):
        v_future = np.dot(self.trans_model[state], self.v)
        q = self.reward_model[state] + self.gamma ** self.nstep * v_future
        return q

    def _eval_policy_returns(self, policy):
        self.domain.set_eval_mode(True, default=False)
        sum_returns = 0
        for ep in range(self.eval_episodes):
            episode_return = 0
            discount_factor = 1
            state = self.domain.get_initial_state()
            for i in range(self.eval_max_steps):
                action = self.select_from_policy(state, policy)
                state, reward = self.domain.take_action(state, action)
                episode_return += discount_factor * reward
                discount_factor *= self.gamma
                if state == -1:
                    sum_returns += episode_return
                    break
        self.domain.set_eval_mode(False)
        return sum_returns / self.eval_episodes

    def eval_policy(self):
        policy = self.get_eval_policy()
        perf = self._eval_policy_returns(policy)
        return perf

    def get_eval_policy(self):
        q = self._get_q()
        return self._get_egreedy_policy(q, self.eval_epsilon)

    def get_current_policy(self):
        q = self._get_q()
        return self._get_egreedy_policy(q, self.epsilon)

    def _get_egreedy_policy(self, q, epsilon):
        policy = np.zeros([self.num_states, self.num_actions])
        for s in range(self.num_states):
            qmax = np.max(q[s])
            max_indices = np.nonzero(q[s] == qmax)[0]
            num_max = max_indices.size
            policy[s] = np.ones(self.num_actions) * epsilon / self.num_actions
            policy[s][max_indices] += (1 - epsilon) / num_max
        return policy

    def select_from_policy(self, state, policy):
        rnd = np.random.random() / 1.0000001
        sum_p = 0.0
        for a in range(self.num_actions):
            sum_p += policy[state][a]
            if rnd < sum_p:
                return a
        assert False, "action not selected"

    def select_action(self, state):
        # selects e-greedy optimal action.
        q = self.compute_q_onestep_rollout(state)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            qmax = np.max(q)
            max_indices = np.nonzero(q == qmax)[0]
            num_max = max_indices.size
            i = np.random.randint(num_max)
            action = max_indices[i]

        return action
