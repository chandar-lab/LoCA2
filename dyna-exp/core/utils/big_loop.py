#######################################################################
# Copyright (C) 2022 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


from ..utils.logger import get_logger
import time
import numpy as np


def run_steps(config):
	config.seed = config.sweep_id * config.num_workers + config.rank
	agent = config.agent_class(config)
	agent_name = agent.__class__.__name__
	if config.rank == 0:
		agent.logger = get_logger(config.exp_name, config.sweep_id, log_level=config.log_level)
		report_string = "\n"
		for k in config.param_sweeper_dict:
			report_string += "%s: %s \n" % (str(k), str(config.param_sweeper_dict[k]))
		agent.logger.info(report_string)
	total_episodes = 0
	reports = {}
	t0 = time.time()
	last_total_steps = 0
	phase = 1
	while True:
		if config.rank == 0:
			report = agent.step()
			if report is not None:
				for key in report:
					reports.setdefault(key, []).extend(report[key])
			# print(agent.total_steps)
			log_flag = int(agent.total_steps / config.log_interval) > int(last_total_steps / config.log_interval)
			save_network_flag = int(agent.total_steps / config.save_interval) > int(
				last_total_steps / config.save_interval)
			eval_flag = int(agent.total_steps / config.eval_interval) > int(last_total_steps / config.eval_interval)
			if log_flag and len(reports) != 0:
				report_string = '\ntotal steps %d\n' % (agent.total_steps)
				for report_name in reports:
					report = reports[report_name]
					if report_name == 'episodic_return':
						total_episodes += len(reports[report_name])
						report_string += 'total episodes %3d\n' % (total_episodes)
					if report_name == 'rewards':
						report_string += 'average reward %3f\n' % (np.sum(report) / config.log_interval)
					if len(report) != 0:
						report_string += report_name + ' %.3f/%.3f/%.3f/%.3f/%d (mean/median/min/max/num)\n' % (
							np.mean(report), np.median(report), np.min(report),
							np.max(report), len(report)
						)
				report_string += '%.3f steps/s\n' % (config.log_interval / (time.time() - t0))
				agent.logger.info(report_string)
				reports = {}
			t0 = time.time()
			if config.if_save_network and save_network_flag:
				agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
			if config.if_eval_episodes and eval_flag:
				eval_reports = agent.eval_episodes()
				report_string = '\ntotal steps %d\n' % (agent.total_steps)
				for report_name in eval_reports:
					report = eval_reports[report_name]
					if len(report) != 0:
						report_string += report_name + '_test %.3f/%.3f/%.3f/%.3f/%d (mean/median/min/max/num)\n' % (
							np.mean(report), np.median(report), np.min(report),
							np.max(report), len(report)
						)
				agent.logger.info(report_string)
			if config.if_eval_steps and eval_flag:
				avg_reward = agent.eval_n_steps()
				report_string = '\naverage_reward_test' + ' %.3f over %d steps\n' % (avg_reward, config.eval_steps)
				agent.logger.info(report_string)
		else:
			agent.step()
		
		if hasattr(config, "phase1_steps") and agent.total_steps >= config.phase1_steps and phase == 1:
			if hasattr(agent, 'set_phase_and_eval') is False:
				assert False, 'agent does not have set_phase_and_eval function'
			agent.set_phase_and_eval(2)
			phase = 2
		
		if hasattr(config, "phase1_steps") and hasattr(config, "phase2_steps") and agent.total_steps >= config.phase1_steps + config.phase2_steps and phase == 2:
			if hasattr(agent, 'set_phase_and_eval') is False:
				assert False, 'agent does not have set_phase_and_eval function'
			agent.set_phase_and_eval(3)
			phase = 3
		
		if agent.total_steps >= config.max_steps:
			agent.close()
			break
		
		last_total_steps = agent.total_steps