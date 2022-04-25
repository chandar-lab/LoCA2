# Dyna Experiments 


## Dependency
* Please check our singularity recipe (LoCA.recipe).

## Usage
```python run.py --config-file <config_file_path> --id <param_setting_id>```

For example:

```python run.py --config-file experiments/linear_dyna/inputs.json --id 0```

Implemented algorithms:
* Sarsa(lambda)
* Linear Dyna
* Nonlinear Dyna Q

Implemented LoCA environments:
* MountainCar
