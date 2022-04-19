# Deep Reinforcement Learning Experiments

Here, we introduce the ReacherLoCA domain and provide the code for 
reproducing the experiments in the paper for the DreamerV2 agent. 

We have provided the ReacherLoCA's installation procedure in the `ReacherLoCA` dir. Furthermore, `DreamerV2` dir 
contains our simple modifications to the [DreamerV2's original implementation](https://github.com/danijar/dreamerv2). 
Since the essential wrappers around the ReacherLoCA domain are provided in the code, configuring other codebases 
to evaluate their adaptivity should be trivial. The only changes required are for the main train loop and adding
the ReacherLoCA domain to the given codebase.