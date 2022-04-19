################################################
#  Author: Harm van Seijen
#  Copyright 2022 Microsoft
################################################




import numpy as np

class SarsaLambdaAgent(object):

    def __init__(self, settings, domain):

        self.domain = domain
        self.alpha = settings['alpha']
        self.lAmbda = settings['lambda']
        self.type = settings['traces_type']
        self.epsilon_default = settings['epsilon']
        self.epsilon = self.epsilon_default
        self.q_init = settings['q_init']
        self.eval_episodes = settings['eval_episodes']
        self.eval_epsilon = settings['eval_epsilon']
        self.eval_max_steps = settings['eval_max_steps']
        self.max_episode_length = settings['max_episode_length']

        self.num_states = domain.get_num_states()
        self.num_actions = domain.get_num_actions()
        self.gamma = domain.get_gamma()
        self.q = None
        self.q_phase1 = None
        self.q_phase2 = None


    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def reset_epsilon(self):
        self.epsilon = self.epsilon_default

    def run_single_task(self, steps, eval_window_size, single_task_id, rnd_explore_steps = 0):
        print("Singe-task run started.")
        self.domain.set_task(single_task_id)  # 0 : A ,   1: B
        self.domain.set_phase(0)  # determines initial state distribution
        #self.domain.act_optimal()
        self.q = np.ones([self.num_states, self.num_actions]) * self.q_init

        perf = self.run(steps, eval_window_size, rnd_explore_steps)

        print("Single-task avg return: ", np.mean(perf), " final return: ", perf[-1])
        return perf

    def run_phase1(self, steps, eval_window_size, rnd_explore_steps = 0):
        print("Phase 1 started.")
        self.domain.set_task(0)  # set domain to task A
        self.domain.set_phase(1)
        self.q = np.ones([self.num_states, self.num_actions]) * self.q_init

        perf = self.run(steps, eval_window_size, rnd_explore_steps)  # HvS: No longer fixed hehavior during transition
        self.q_phase1 = np.copy(self.q)

        print("Phase 1: avg return: ", np.mean(perf), " final return: ", perf[-1])
        return perf


    def run_phase2(self, steps, eval_window_size):
        print("Phase 2 started.")
        self.domain.set_task(1)  # set domain to task B
        self.domain.set_phase(2)
        assert self.q_phase1 is not None
        self.q = np.copy(self.q_phase1)

        perf = self.run(steps, eval_window_size)
        self.q_phase2 = np.copy(self.q)

        print("Phase 2: avg return: ", np.mean(perf), " final return: ", perf[-1])
        return perf


    def run_phase3(self, steps, eval_window_size):
        print("Phase 3 started.")
        self.domain.set_task(1)  # set domain to task B
        self.domain.set_phase(3)
        assert self.q_phase2 is not None
        self.q = np.copy(self.q_phase2)

        perf = self.run(steps, eval_window_size)

        print("Phase 3: avg return: ", np.mean(perf), " final return: ", perf[-1])
        return perf


    def run(self, steps, window_size, rnd_explore_steps  = 0):
        # perform a run over certain number of time steps

        num_datapoints = steps // window_size
        data_point = 0
        self.epsilon = 1.0

        performance = np.zeros([num_datapoints])
        performance[data_point] = self.eval_policy()
        data_point += 1

        next_state = -1
        for i in range(steps):

            if i >= rnd_explore_steps:
                self.epsilon = self.epsilon_default

            if next_state == -1 or time == self.max_episode_length:
                time = 0
                state = self.domain.get_initial_state()
                self._reset_trace()
                action = self.select_action(state)
            else:
                state = next_state
                action = next_action

            #action = self.select_action(state)  # HvS: remove this line - Q-learning thingy
            next_state, reward = self.domain.take_action(state, action);  time += 1
            next_action = self.select_action(next_state)


            # perform update
            self._perform_update(state, action, reward, next_state, next_action)


            if ((i+1) % window_size == 0) & (data_point < num_datapoints):
                performance[data_point] = self.eval_policy()
                data_point += 1

        return performance

    def _reset_trace(self):
        self.e_trace = np.zeros([self.num_states, self.num_actions])
        self.q_sa_old = 0

    def _perform_update(self, state, action, reward, next_state, next_action):

        q_sa = self.q[state][action]
        if next_state == -1:
            q_sa_next = 0
        else:
            q_sa_next = self.q[next_state][next_action]

        if self.type == 2:
            delta = reward + self.gamma * q_sa_next - self.q_sa_old
        else:
            delta = reward + self.gamma * q_sa_next - q_sa

        # update traces
        if self.type == 0:  # accumulating traces
            self.e_trace[state][action] += self.alpha
        elif self.type == 1:  # replacing traces
            self.e_trace[state][action] = self.alpha
        elif self.type == 2:  # dutch traces
            e_phi = self.e_trace[state][action]
            self.e_trace[state][action] += self.alpha * (1 - e_phi)
        else:
            assert False

        self.q += self.e_trace * delta

        if self.type == 2:
            self.q[state][action] -= self.alpha * (q_sa - self.q_sa_old)
            self.q_sa_old = q_sa_next

        self.e_trace *= self.gamma * self.lAmbda


    def _eval_policy_returns(self, policy):
        self.domain.set_eval_mode(True, default=False)
        sum_returns = 0
        for ep in range(self.eval_episodes):
            R = 0
            discount_factor = 1
            state = self.domain.get_initial_state()
            for i in range(self.eval_max_steps):
                action = self.select_from_policy(state, policy)
                state, reward = self.domain.take_action(state, action)
                R += discount_factor * reward
                discount_factor *= self.gamma
                if state == -1:
                    sum_returns += R
                    break
        self.domain.set_eval_mode(False)
        return sum_returns / self.eval_episodes


    def eval_policy(self):
        policy = self.get_eval_policy()
        perf = self._eval_policy_returns(policy)
        return perf


    def get_eval_policy(self):
        return self._get_egreedy_policy(self.q, self.eval_epsilon)

    def get_current_policy(self):
        return self._get_egreedy_policy(self.q, self.epsilon)

    def _get_egreedy_policy(self, q, epsilon):
        policy = np.zeros([self.num_states, self.num_actions])
        for s in range(self.num_states):
            qmax = np.max(q[s])
            max_indices = np.nonzero(q[s] == qmax)[0]
            num_max = max_indices.size
            policy[s] = np.ones(self.num_actions) * epsilon / self.num_actions
            policy[s][max_indices] += (1 - epsilon) / num_max
        return policy


    def select_from_policy(self,state, policy):
        rnd = np.random.random()/1.0000001
        sum_p = 0.0
        for a in range(self.num_actions):
            sum_p += policy[state][a]
            if rnd < sum_p:
                return a
        assert False, "action not selected"


    def select_action(self,state):
        # selects e-greedy optimal action.
        if np.random.random() < self.epsilon:
            action= np.random.randint(self.num_actions)
        else:
            qmax = np.max(self.q[state])
            max_indices = np.nonzero(self.q[state] == qmax)[0]
            num_max = max_indices.size
            i = np.random.randint(num_max)
            action = max_indices[i]

        return action