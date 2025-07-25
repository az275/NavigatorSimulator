
# Navigator: scheduler and object placement simulator
<br />

This Repository contains the Simulation code to run Navigator on Customized workflow and cluster.

##  Run simulation
### 1. set environment variable at the terminal 
#### * For Unix/Linux

`source set_env.sh` OR

export SIMULATION_DIR=/path/to/simulation/directory
export PYTHONPATH="${PYTHONPATH}:{SIMULATION_DIR}"
#### * For Windows
set PYTHONPATH=%PYTHONPATH%;C:\path\to\directory\

### 2. Simulatioin

#### 2.1 Set Config and Workflow
User can set customized system configuration at /core/config.py and /core/workflow.py.

#### 2.2 Run simulation
Run the python file, /experiments/run_experiments.py , to start the simulation and generate logging data at ./experiment

``` python ./run_experiments.py <experiment_scheduler0> <experiment_scheduler1> ..(centralheft|decentralheft|hashtask) ```
