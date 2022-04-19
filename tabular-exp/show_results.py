################################################
#  Author: Harm van Seijen
#  Copyright 2022 Microsoft
################################################




import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

single_task = False

filenames = {};
labels = {}; i = 0


if single_task:
    i += 1; filenames[i] = 'mb_1step_random_single_task_tmp'; labels[i] = 'mb-1-r';
else:
    i += 1; filenames[i] = 'sarsa_lambda_stoch025'; labels[i] = 'Sarsa($\lambda$)';
    i += 1; filenames[i] = 'mb_1step_random_stoch025'; labels[i] = 'mb-1-r';
    i += 1; filenames[i] = 'mb_1step_current_stoch025'; labels[i] = 'mb-1-c';
    i += 1; filenames[i] = 'mb_2_step_random_stoch025'; labels[i] = 'mb-2-r';


results = {}
time = {}
num_results = i


for i in range(1, num_results+1):

    performance = np.load('data/' + filenames[i] + '_results.npy')
    results[i] = np.mean(performance, axis=0)

    with open('data/' + filenames[i] + '_settings.txt') as f:
        settings = json.load(f)
        window_size = settings['eval_window_size']
        if single_task:
            single_task_id = settings['single_task_id']
            num_single_task_steps = settings['num_single_task_steps']
        else:
            num_phase1_steps = settings['num_phase1_steps']
        num_datapoints = results[i].size
        time[i] = np.arange(0, num_datapoints) * window_size / 1000

# # optimal policy, stoch = 0
# optimal_taskA = 2.849
# optimal_taskB = 1.421

# optimal policy, stoch = 0.25
optimal_taskA = 2.525
optimal_taskB = 1.257



if single_task:
    if single_task_id == 0:
        results[0] = np.ones(results[num_results].size) * optimal_taskA
    else:
        results[0] = np.ones(results[num_results].size) * optimal_taskB
else:
    results[0]  = np.ones(results[num_results].size)*optimal_taskA
    results[0][num_phase1_steps//window_size:] = optimal_taskB
labels[0] = 'optimal'
time[0] = time[num_results]
num_results += 1

##########
plt.figure(figsize=(10,4))

font_size = 20
font_size_legend = 20
font_size_title = 20

top = 0.930
bottom = 0.145
left = 0.170
right = 0.960

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = '15'
plt.rcParams['text.usetex'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

plt.rc('font', size=font_size)  # controls default text sizes
plt.rc('axes', titlesize=font_size_title)  # fontsize of the axes title
plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
plt.rc('ytick', labelsize=font_size)  # fontsize of the lt.rc('legend', fontsize=font_size_legend)  # legend fontsize
plt.rc('figure', titlesize=font_size)  # fontsize of the figure title
plt.axes((left,bottom,right-left,top-bottom))

#plt.plot(time[0], results[0], 'k-', marker='', label=labels[0])
for i in range(2,num_results):
   plt.plot(time[i],results[i],marker = '', label=labels[i])
plt.plot(time[1], results[1],  marker='', label=labels[1])
plt.plot(time[0], results[0], 'k-', marker='', label=labels[0])

if not single_task:
    plt.plot([100, 100],[0, 10],'k:')
    plt.plot([150, 150], [0, 1.5], 'k:')
    plt.xlim(0, 200)
    plt.ylim(0,2.75)
    plt.title('')
    y_text = 0.1
    plt.text(40, y_text, 'Phase 1')
    plt.text(115, y_text, 'Phase 2')
    plt.text(165, y_text, 'Phase 3')
    plt.ylabel('avg. return')
    plt.legend(loc=0,ncol=2)

plt.xlabel('time steps (x 1000)')
plt.show()