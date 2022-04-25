# Tabular Experiments

## Dependency
```sh
pip install numpy matplotlib 
```

## Usage
To reproduce the results from the paper:
1) Change the directory to the LoCA2_tabular:```cd LoCA2_tabular```.
2) In ```main.py```  set the variable `method` :

|variable assignment | method|
|:-------------|:-------------|
| `method  = 1`    |  sarsa(lambda)|
| `method  = 2`    |  mb_1step_random|
| `method  = 3`    |  mb_1step_current|
| `method  = 4`    |  mb_2_step_random|
4) Train the selected method, using:```python main.py ```. It takes between 2-5 minutes to train a method. Results are written to /data
5) After training all 4 methods, visualize the results using:  ```python show_results.py```.



