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


## Usage
For ease of use, we have divided the structure of the code into three directories. Each directory includes the code for a specific section of the paper:
- `tabular-exp`: Section 3
- `deeprl-exp`: Section 4
- `dyna-exp`: Section 5

Please follow the instructions of each directory to be able to reproduce the results presented in the paper.

Note: The code for each section has been tested with `python 3.7`. 

## Citation

If you found this work useful, please consider citing [our paper](https://arxiv.org/abs/2204.11464) 
and [the previous work](https://arxiv.org/abs/2007.03158). 

```
@article{wan2022loca2,
  title={Towards Evaluating Adaptivity of Model-Based Reinforcement Learning Methods},
  author={Wan, Yi and Rahimi-Kalahroudi, Ali and Rajendran, Janarthanan and Momennejad, Ida and Chandar, Sarath and van Seijen, Harm},
  journal={arXiv preprint arXiv:2204.11464},
  year={2022}
}
```

```
@misc{seijen2020loca,
    title={The LoCA Regret: A Consistent Metric to Evaluate Model-Based Behavior in Reinforcement Learning},
    author={Harm van Seijen and Hadi Nekoei and Evan Racah and Sarath Chandar},
    year={2020},
    eprint={2007.03158},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
