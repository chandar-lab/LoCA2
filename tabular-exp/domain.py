################################################
#  Author: Harm van Seijen
#  Copyright 2022 Microsoft
################################################





import numpy as np
import math


class Domain:
    #
    # Domain
    # Navigation task with on either side a terminal state.
    # Example, width = 4, height = 2:
    #
    #    y=    0 1 2 3
    #  x=0  -1 0 1 2 3 -1
    #    1  -1 4 5 6 7 -1
    #
    #  (-1 indicates terminal state)
    #
    #  Action:
    #      a = 0  up
    #      a = 1  right
    #      a = 2  down
    #      a = 3  left
    #
    #  stoch = 0.2 means with probability 0.2 environment generates random action outcome instead of intended one
    #
    #  reward_T2 : reward received upon entering left terminal state; reward_T1: reward for right terminal state
    #  task 0:   reward_T2 = 2,  reward_T1 = 4
    #  task 1:   reward_T2 = 2,  reward_T1 = 1
    #



    def __init__(self, settings):
        self.task = 0       # domain defaults to task A
        self.gamma = settings['gamma']
        self.width = settings['width']
        self.height = settings['height']
        self.stoch = settings['stochasticity']

        self.init_state_single_task = settings['init_state_single_task']
        self.init_state_phase1 = settings['init_state_phase1']
        self.init_state_phase2 = settings['init_state_phase2']
        self.init_state_phase3 = settings['init_state_phase3']
        self.init_state_eval = settings['init_state_eval']
        self.init_state = None

        self.use_optimal_policy = False

        self.num_actions = 4
        self.num_states = self.width*self.height  # note: state-space-multiplier not included

        self.reward_taskA_T2 = 2
        self.reward_taskA_T1 = 4

        self.reward_taskB_T2 = 2
        self.reward_taskB_T1 = 1
        return

    def set_task(self, task):
        self.task = task
        if task == 0:    # taskA
            self.reward_T1 = self.reward_taskA_T1
            self.reward_T2 = self.reward_taskA_T2
        elif task == 1:   # taskB
            self.reward_T1 = self.reward_taskB_T1
            self.reward_T2 = self.reward_taskB_T2
        else:
            assert False, 'incorrect task identifier'




    def set_eval_mode(self, eval, default=True):
        if eval:
            if default:
                if self.reward_T1 > self.reward_T2:
                    self.reward_T1 = 1
                    self.reward_T2 = 0
                else:
                    self.reward_T1 = 0
                    self.reward_T2 = 1
            self.init_state = self.init_state_eval
        else:
            self.set_task(self.task)
            self.set_phase(self.phase)

    def get_initial_state(self):
        # self.init_state specifies y-values that cover init state area
        # self.init_state = -1 means the whole state-space is drawn from
        if isinstance(self.init_state,np.ndarray):
            i = np.random.randint(len(self.init_state))
            init_y = self.init_state[i]
            init_x = np.random.randint(self.height)
            initial_state = init_x*self.width + init_y
        else:
            initial_state = np.random.randint(self.num_states)
        assert 0 <= initial_state < self.num_states

        return initial_state

    def set_phase(self, phase):
        self.phase = phase      # phase = 1, 2, 3 OR phase = 0 if single_task experiment
        if phase == 1:
            self.init_state = self.init_state_phase1
        elif phase == 2:
            self.init_state = self.init_state_phase2
        elif phase == 3:
            self.init_state = self.init_state_phase3
        elif phase == 0:
            self.init_state = self.init_state_single_task
        else:
            assert False, 'incorrect identifier'


    def get_gamma(self):
        return self.gamma

    def get_task(self):
        return self.task

    def get_num_states(self):
        return self.num_states

    def get_num_actions(self):
        return self.num_actions

    def act_optimal(self):
        self.use_optimal_policy = True
        q = self._compute_optimal_q(epsilon=0)
        self.optimal_policy = np.argmax(q,axis=1)

    def take_action(self, state, action):
        assert (state >= 0)
        assert (state < self.num_states)
        assert (action < self.num_actions)

        if self.use_optimal_policy:
            action = self.optimal_policy[state]  # overwrite with optimal action

        if np.random.random() < self.stoch:
            action = np.random.randint(self.num_actions)

        next_state, reward = self._take_det_action(state, action)
        return next_state, reward


    def _take_det_action(self, state, action):
        x = state // self.width
        y = state % self.width

        reward = 0  # default reward
        if action == 0:   # up
            x_new = max(x-1,0)
            y_new = y
        elif action == 1:  # right
            if y==self.width-1:
                reward = self.reward_T1
                return -1, reward
            x_new = x
            y_new = min(y+1,self.width-1)
        elif action == 2:  # down
            x_new = min(x+1,self.height-1)
            y_new = y
        else:  # left
            if y==0:
                reward = self.reward_T2
                return -1, reward
            x_new = x
            if y==self.width-1:     # New: This implements not going back right before terminal on the right
                y_new = y
            else:
                y_new = max(y - 1, 0)

        next_state = x_new*self.width + y_new

        return next_state, reward


    def _compute_model(self):
        # Computes transition function P(s,a,s')

        expected_reward = np.zeros([self.num_states, self.num_actions])
        transition_tensor = np.zeros([self.num_states, self.num_actions,  self.num_states])

        for state in range(self.num_states):
            transitions_local = np.zeros([self.num_actions, self.num_states])
            reward_local = np.zeros([self.num_actions])
            for action in range(self.num_actions):
                next_state, reward = self._take_det_action(state,action)

                reward_local[action] = reward
                if next_state != -1:  # if not terminal state
                    transitions_local[action, next_state] = 1
            transitions_mean = np.mean(transitions_local, axis = 0)
            reward_mean = np.mean(reward_local, axis=0)


            for action in range(self.num_actions):
                stoch_transitions = self.stoch*transitions_mean + (1-self.stoch)*transitions_local[action]
                transition_tensor[state, action, :] = stoch_transitions
                stoch_reward = self.stoch*reward_mean + (1-self.stoch)*reward_local[action]
                expected_reward[state, action] = stoch_reward

        return transition_tensor, expected_reward


    def _compute_optimal_q(self, epsilon):
        iter = 100

        transition_tensor, expected_reward = self._compute_model()

        q = np.zeros([self.num_states, self.num_actions])

        for i in range(iter):
            v = np.mean(q, axis=1)*epsilon + np.max(q, axis=1)*(1-epsilon)
            v_next = np.dot(transition_tensor, v)
            q = expected_reward + self.gamma*v_next

        return(q)



    def print_optimal_action(self, epsilon=0):

        self._print_optimal_action(0)
        if epsilon > 0:
            self._print_optimal_action(epsilon)

    def show_policy(self,policy):

        a_grid = np.zeros([self.height, self.width],dtype=np.int8)
        for s in range(self.num_states):
            x = s // self.width
            y = s % self.width

            a_grid[x][y] = np.argmax(policy[s])

        print('policy:')
        # print optimal action
        for y in range(self.width):
            print('{:3s}'.format(str(y)), end='')
        print('')
        for x in range(self.height):
            for y  in range(self.width):
                a = a_grid[x][y]
                dir = 'URDL'
                print('{:3s}'.format(dir[a]),end='')
            print('')

    def show_value(self, v):
        v_grid = np.zeros([self.height, self.width])
        for s in range(self.num_states):
            x = s // self.width
            y = s % self.width
            v_grid[x][y] = v[s]

        print('value:')
        for x in range(self.height):
            for y  in range(self.width):
                v = v_grid[x][y]
                print('{:3.3f}  '.format(v),end='')
            print('')

    def _get_egreedy_policy(self, q, epsilon):
        policy = np.zeros([self.num_states, self.num_actions])
        for s in range(self.num_states):
            qmax = np.max(q[s])
            max_indices = np.nonzero(q[s] == qmax)[0]
            num_max = max_indices.size
            policy[s] = np.ones(self.num_actions) * epsilon / self.num_actions
            policy[s][max_indices] += (1 - epsilon) / num_max
        return policy

    def _eval_policy(self, policy):

        iter = 100

        transition_tensor, expected_reward = self._compute_model()

        q = np.zeros([self.num_states, self.num_actions])

        for i in range(iter):
            v = np.sum(np.multiply(policy, q), axis=1)
            v_next = np.dot(transition_tensor, v)
            q = expected_reward + self.gamma * v_next

        return(q)



    def _print_optimal_action(self, epsilon):

        q = self._compute_optimal_q(epsilon)
        a_optimal = np.argmax(q,axis=1)
        q_mean = np.mean(q,axis=1)
        q_max = np.max(q,axis=1)
        v_optimal = (1-epsilon)*q_max + epsilon*q_mean

        a_grid = np.zeros([self.height, self.width],dtype=np.int8)
        v_grid = np.zeros([self.height, self.width])
        for s in range(self.num_states):
            x = s // self.width
            y = s % self.width

            a_grid[x][y] = a_optimal[s]
            v_grid[x][y] = v_optimal[s]

        print('')
        if epsilon == 0:
            print('optimal v:')
        else:
            print('{:1.1f}-greedy-optimal v'.format(epsilon))
        # print optimal state value
        for x in range(self.height):
            for y  in range(self.width):
                v = v_grid[x][y]
                print('{:3.3f}  '.format(v),end='')
            print('')

        print('')
        if epsilon == 0:
            print('optimal a:')
        else:
            print('{:1.1f}-greedy-optimal a'.format(epsilon))
        # print optimal action
        for y in range(self.width):
            print('{:3s}'.format(str(y)), end='')
        print('')
        for x in range(self.height):
            for y  in range(self.width):
                a = a_grid[x][y]
                dir = 'URDL'
                print('{:3s}'.format(dir[a]),end='')
            print('')



def test():
    settings = {}
    settings['gamma'] = 0.97
    settings['width'] = 25
    settings['height'] = 4
    settings['stochasticity'] = 0

    domain = Domain(settings)


    epsilon = 0.1
    domain.set_task(0)
    q1 = domain._compute_optimal_q(0)
    policy1 = domain._get_egreedy_policy(q1, epsilon)
    domain.show_policy(policy1)
    #domain.show_value(np.max(q1,axis=1))


    domain.set_task(1)
    q2 = domain._eval_policy(policy1)
    policy2 = domain._get_egreedy_policy(q2, epsilon)
    domain.show_policy(policy2)
    #domain.show_value(np.max(q2, axis=1))


if __name__ == "__main__":
    test()
