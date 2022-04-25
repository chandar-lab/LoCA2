# Towards Evaluating Adaptivity of MBRL Methods

Official code for the "Towards Evaluating Adaptivity of Model-Based Reinforcement Learning" paper.

----

## Abstract 
In recent years, a growing number of deep model-based reinforcement learning (RL) methods have been introduced. 
The interest in deep model-based RL is not surprising, given its many potential benefits, such as higher 
sample efficiency and the potential for fast adaption to changes in the environment. However, we demonstrate, 
using an improved version of the recently introduced Local Change Adaptation (LoCA) evaluation methodology, 
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

## Citation

If you found this work useful, please consider citing our paper. 

