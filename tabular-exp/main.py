################################################
#  Author: Harm van Seijen
#  Copyright 2022 Microsoft
################################################



import numpy as np
import time
import json
from domain import Domain
from sarsa_lambda import SarsaLambdaAgent
from modelbased import ModelBasedAgent



# Settings  ##########################################################################


# METHODS:
# 1:  sarsa_lambda
# 2:  mb_1step_random
# 3:  mb_1step_current
# 4:  mb_2_step_random


# Experiment settings
method = 4    # See above more methods

single_task = False   # if True: only task A or B is evaluated;  if False, full experiment is performed
single_task_id = 1  # 0 : taskA,  1 : taskB   #only relevant if single task.
num_single_task_steps = 100000 #100000   # only relevant if single_task == True
num_phase1_steps = 100000
num_phase2_steps = 50000
num_phase3_steps = 50000
rnd_explore_steps = 0

eval_window_size = 500    # evaluation occurs every 'eval_window_size' steps

num_runs = 10
filename_extension = '_stoch025'

if method == 1:
    method_name = 'sarsa_lambda'
elif method == 2:
    method_name = 'mb_1step_random'
elif method == 3:
    method_name = 'mb_1step_current'
elif method == 4:
    method_name = 'mb_2_step_random'
else:
    assert False, 'HvS: Invalid method id.'

# Domain settings
domain_settings = {}
domain_settings['height'] = 4
domain_settings['width'] = 25
domain_settings['stochasticity'] = 0.25
domain_settings['gamma'] = 0.97
domain_settings['init_state_single_task'] = -1
domain_settings['init_state_phase1'] = -1
domain_settings['init_state_phase2'] = np.array([24])
domain_settings['init_state_phase3'] = -1
domain_settings['init_state_eval'] = -1




# Agent settings
agent_settings = {}
agent_settings = {}
if method == 1:   # Sarsa(lambda)
    agent_settings['alpha'] = 0.03    # traces requires lower step-size
else:
    agent_settings['alpha'] = 0.2
agent_settings['alpha_v'] = 1
agent_settings['lambda'] = 0.95  # only relevant for sarsa_lambda
agent_settings['method'] = method_name
agent_settings['max_episode_length'] = 100
agent_settings['traces_type'] = 2   # only relevant for sarsa_lambda  -- 0: accumulating traces, 1; replacing traces, 2: dutch traces
agent_settings['epsilon'] = 0.1
agent_settings['q_init'] = 8
agent_settings['eval_episodes'] = 100
agent_settings['eval_epsilon'] = 0
agent_settings['eval_max_steps'] = 40

#############################################################################################


if single_task  == True:
    filename = method_name + '_single_task' + filename_extension
else:
    filename = method_name + filename_extension


print("file: ", filename)


#############################################################################################


my_domain = Domain(domain_settings)
if method == 1:
    my_agent = SarsaLambdaAgent(agent_settings, my_domain)
else:
    my_agent = ModelBasedAgent(agent_settings, my_domain)

start = time.time()
if single_task:
    data_size = num_single_task_steps // eval_window_size
    performance = np.zeros([num_runs, data_size])
    for run in range(num_runs):
        print('');
        print("### run: ", run, " ############################")
        performance[run, :] = my_agent.run_single_task(num_single_task_steps, eval_window_size, single_task_id)
else:
    data_size1 = num_phase1_steps // eval_window_size
    data_size2 = num_phase2_steps // eval_window_size
    data_size3 = num_phase3_steps // eval_window_size
    performance = np.zeros([num_runs, data_size1 + data_size2 + data_size3])
    for run in range(num_runs):
        print(''); print("### run: ", run, " ############################")
        performance[run, :data_size1] = my_agent.run_phase1(num_phase1_steps, eval_window_size, rnd_explore_steps)
        performance[run, data_size1:data_size1 + data_size2] = my_agent.run_phase2(num_phase2_steps, eval_window_size)
        performance[run, data_size1 + data_size2 :] = my_agent.run_phase3(num_phase3_steps, eval_window_size)
end = time.time()

avg_performance = np.mean(performance,axis=0)
print('')
print("time: {}s".format(end-start))
print("avg return: ", np.mean(avg_performance), " final return: ", avg_performance[-1])


if single_task:
    num_points = avg_performance.size
    num_buckets = 10
    window = num_points // num_buckets
    if single_task_id == 0:
        print("Task A. ",end='')
    else:
        print("Task B. ", end='')
    for i in range(num_buckets):
        bucket_perf = np.mean(avg_performance[window*i:window*(i+1)])
        print(" {:3.2f}".format(bucket_perf), end='')
    print('')

    avg_perf2 = np.mean(avg_performance[num_points//2:])
    print("avg perf, 2nd half: {:3.3f}".format(avg_perf2))

else:
    perf_avg = np.mean(performance)
    perf_final = np.mean(performance[:, -1])
    print("avg. perf. : {:3.2f}".format(perf_avg), ", std error: {:3.2f}".format(perf_final))



# Store results + some essential settings
settings = {}
settings['method_name'] = method_name
settings['single_task'] = single_task
if single_task:
    settings['single_task_id'] = single_task_id
    settings['num_single_task_steps'] = num_single_task_steps
else:
    settings['num_phase1_steps'] = num_phase1_steps
    settings['num_phase2_steps'] = num_phase2_steps
    settings['num_phase3_steps'] = num_phase3_steps
    settings['rnd_explore_steps'] = rnd_explore_steps
settings['eval_window_size'] = eval_window_size
settings['num_runs'] = num_runs
settings['stochasticity'] = domain_settings['stochasticity']
settings['alpha'] = agent_settings['alpha']
settings['lambda']  = agent_settings['lambda']
settings['q_init'] = agent_settings['q_init']

print("file: ", filename)

with open('data/' + filename + '_settings.txt', 'w') as json_file:
    json.dump(settings, json_file)
np.save('data/' +  filename + '_results.npy', performance)


print('Done.')