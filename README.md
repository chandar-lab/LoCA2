# Towards Evaluating Adaptivity of MBRL Methods

Official code for the "Towards Evaluating Adaptivity of Model-Based Reinforcement Learning" [paper](https://arxiv.org/abs/2204.11464).

----

## Abstract 
In recent years, a growing number of deep model-based reinforcement learning (RL) methods have been introduced. 
The interest in deep model-based RL is not surprising, given its many potential benefits, such as higher 
sample efficiency and the potential for fast adaption to changes in the environment. However, we demonstrate, 
using an improved version of the recently introduced 
[Local Change Adaptation (LoCA) evaluation methodology](https://arxiv.org/abs/2007.03158), 
that well-known model-based methods such as PlaNet and DreamerV2 perform poorly in their ability to adapt to 
local environmental changes.  Combined with prior work that made a similar observation about the other popular 
model-based method, MuZero, a trend appears to emerge, suggesting that current deep model-based methods have serious 
limitations. We dive deeper into the causes of this poor performance, by identifying elements that hurt adaptive 
behavior and linking these to underlying techniques frequently used in deep model-based RL. We empirically validate 
these insights in the case of linear function approximation by demonstrating that a modified version of linear Dyna 
achieves effective adaptation to local changes. Furthermore, we provide detailed insights into the challenges of 
building an adaptive nonlinear model-based method, by experimenting with a nonlinear version of Dyna.

## Setup

<p align="center">
<img src="https://user-images.githubusercontent.com/79111421/165789118-fcd43524-be8e-4e3c-b4dd-1a7d6ce08a93.png" width=40%>
</p>

An experiment in our improved version of the previously introduced LoCA setup consists of 
three different training phases. During Phase 1, the reward function is rA; upon transitioning 
to Phase 2, the reward function changes to rB and remains unchanged upon transitioning to Phase 3. 
Crucially, the initial state distribution during training differs for the different phases (see the figure above). 
We evaluate performance by simply measuring the average return over the evaluation episodes and comparing 
it with the average return of the corresponding optimal policy during all phases. Furthermore, as initial-state distribution, 
the full state-space is used.

Under our new experiment configuration, we call a method adaptive if it is able to reach (near) optimal expected 
return in Phase 2 (after sufficiently long training) while also reaching (near) optimal expected return in Phase 1. 
If a method is able to reach (near) optimal expected return in Phase 1 but not in Phase 2, we call the method non-adaptive.

____ 

Additionally, we have presented a continuous-action domain with pixel-level states that enabled us to 
evaluate [PlaNet](https://arxiv.org/abs/1811.04551) and [DreamerV2](https://arxiv.org/abs/2010.02193) 
using our modified version of the LoCA setup. This domain is a variant of the original Reacher environment 
in the [DeepMind Control Suite](https://github.com/deepmind/dm_control) (see figure below). Furthermore,
we have provided the full instructions on how to use it.

<p align="center">
<img src="https://user-images.githubusercontent.com/79111421/165799596-b6f12157-0996-45be-96f2-b4ecf964c00f.jpg" width=40%>
</p>


## Usage
For ease of use, we have divided the structure of the code into three directories. Each directory includes the code for a specific section of the paper:
- `tabular-exp`: Section 3
- `deeprl-exp`: Section 4
- `dyna-exp`: Section 5

Please follow the instructions of each directory to be able to reproduce the results presented in the paper.

Note: The code for each section has been tested with `python 3.7`. 

## Citation

If you found this work useful, please consider citing the following two LoCA papers:

```
@article{wan2022loca2,
  title={Towards Evaluating Adaptivity of Model-Based Reinforcement Learning Methods},
  author={Wan, Yi and Rahimi-Kalahroudi, Ali and Rajendran, Janarthanan and Momennejad, Ida and Chandar, Sarath and van Seijen, Harm},
  journal={arXiv preprint arXiv:2204.11464},
  year={2022}
}
```


```
@inproceedings{vanseijen-LoCA,
 author = {Van Seijen, Harm and Nekoei, Hadi and Racah, Evan and Chandar, Sarath},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {6562--6572},
 publisher = {Curran Associates, Inc.},
 title = {The LoCA Regret: A Consistent Metric to Evaluate Model-Based Behavior in Reinforcement Learning},
 url = {https://proceedings.neurips.cc/paper/2020/file/48db71587df6c7c442e5b76cc723169a-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
