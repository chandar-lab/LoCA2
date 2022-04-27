# ReacherLoCA Domain

<p align="center">
<img src="https://user-images.githubusercontent.com/79111421/165625119-de96e3a3-b8c9-4da9-843c-f0f898e3c8b3.jpg" width=35%>
</p>

Creating the ReacherLoCA domain is fairly simple as it is very similar to the Reacher domain.  
Starting from `reacher.py` and `reacher.xml` available in the 
[dm_control suite](https://github.com/deepmind/dm_control/tree/main/dm_control/suite), 
we have modified these files to define the new physics and task for the ReacherLoCA domain. 
Any other implementation that fixes the target positions and defines the same dynamics can also be used.

## Installation

In order to use the ReacherLoCA domain through the dm_control suite:
1. Add `reacherloca.py` and `reacherloca.xml` under `dm_control/suite` path in the 
   [dm_control](https://github.com/deepmind/dm_control).
2. Add `from dm_control.suite import reacherloca` to imports of the `dm_control/suite/__init__.py`.
3. Install the `dm_control` manually.

## Usage

To initialize the ReacherLoCA domain:

`env = suite.load('reacherloca', 'easy')`

To change the LoCA task from A to B:

`env.switch_loca_task()`
