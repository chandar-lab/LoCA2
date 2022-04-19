# Evaluating Adaptivity of the DreamerV2 Agent

Using the ReacherLoca domain, we have modified the 
[authors' code]() for the
[DreamerV2](https://arxiv.org/abs/2010.02193) agent in a way that we can 
evaluate the agent's adaptivity using the SimplifiedLoCA notions.


## Usage

As mentioned, this code relies on the original implementation of the DreamerV2 agent by its authors. 
The only files that have changed are `dreamerv2/train.py`, `dreamerv2/common/envs.py`, `dreamerv2/configs.yaml`. 

We decided to run each phase separately in our modification design, 
knowing the fact that the codebase has a great checkpointing procedure.

To run the agent without resetting the replay buffer:
```angular2html
python dreamerv2/train.py --logdir ./logdir --configs loca_dmc_vision --loca_phase phase_1 --steps 50000 --eval_eps 5 --loca_same_dir 1
python dreamerv2/train.py --logdir ./logdir --configs loca_dmc_vision --loca_phase phase_2 --steps 25000 --eval_eps 5 --loca_same_dir 1 
python dreamerv2/train.py --logdir ./logdir --configs loca_dmc_vision --loca_phase phase_3 --steps 50000 --eval_eps 5 --loca_same_dir 1
```

To run the agent while resetting the replay buffer when switching between phases:
```angular2html
python dreamerv2/train.py --logdir ./logdir --configs loca_dmc_vision --loca_phase phase_1 --steps 50000 --eval_eps 5
python dreamerv2/train.py --logdir ./logdir --configs loca_dmc_vision --loca_phase phase_2 --steps 25000 --eval_eps 5 
python dreamerv2/train.py --logdir ./logdir --configs loca_dmc_vision --loca_phase phase_3 --steps 50000 --eval_eps 5
```
In our evaluation in the paper, we have done a grid search over the critical hyperparameters: 
1) `--actor_ent`
2) `--discount`
3) `--loss_scales.kl`

The `common/plot.py` can be used to make the plots for the agent's performance in each phase. 
Additionally, one can write a script to merge the performance logs for all 3 phases and then 
plot them together in just one figure.

## Tips

It's worth looking at the `LOCADMC` class in `common/envs.py`. It contains all the wrappers around 
the ReacherLoCA domain such that the agent can interact with the environment in different phases correctly.