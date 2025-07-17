import os
import sys

from schedulers.centralized.simulation_central import *
from schedulers.decentralized.simulation_decentral import *

from bidict import bidict


sys.dont_write_bytecode = True




# experiment_schedulers options: centralheft | decentralheft | hashtask
experiment_schedulers = []
plotting_job_type_list = [0]
# plotting_job_type_list = [2,3]
np.random.seed(42)

produce_breakdown = True

# Scheduler options
NO_SCHEDULER = 0
DECENTRALHEFT = 1
CENTRALHEFT = 2
HASHTASK = 3
SHEPHERD = 4

SCHEDULER_NAMES = bidict({
    NO_SCHEDULER: "no_scheduler",
    DECENTRALHEFT: "decentralheft",
    CENTRALHEFT: "centralheft",
    HASHTASK: "hashtask",
    SHEPHERD: "shepherd"
})

def run_experiment(scheduler_type: int):
    assert(scheduler_type in [NO_SCHEDULER, DECENTRALHEFT, CENTRALHEFT, HASHTASK, SHEPHERD])

    out_path = os.path.join("results", SCHEDULER_NAMES[scheduler_type])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Created directory at {out_path}")
    else:
        print(f"Directory at {out_path} exists")

    sim = None
    if scheduler_type == CENTRALHEFT:
        sim = Simulation_central(simulation_name="centralheft", job_split="PER_TASK",
                                 num_workers=TOTAL_NUM_OF_NODES, job_types_list=plotting_job_type_list,
                                 produce_breakdown=True)
    elif scheduler_type == DECENTRALHEFT:
        sim = Simulation_decentral(simulation_name="decentralheft", job_split="PER_TASK",
                                   num_workers=TOTAL_NUM_OF_NODES, job_types_list=plotting_job_type_list,
                                   dynamic_adjust=False, consider_load=True, consider_cache=True, produce_breakdown=True)
    elif scheduler_type == HASHTASK:
        sim = Simulation_central(simulation_name="hashtask", job_split="PER_TASK",
                                 num_workers=TOTAL_NUM_OF_NODES, job_types_list=plotting_job_type_list,
                                 produce_breakdown=True)
    elif scheduler_type == SHEPHERD:
        sim = Simulation_central(simulation_name="shepherd", job_split="PER_TASK",
                                 num_workers=TOTAL_NUM_OF_NODES, job_types_list=plotting_job_type_list,
                                 produce_breakdown=True)

    sim.run()

    event_log = sim.event_log
    event_log.to_csv(os.path.join(out_path, "events_by_time.csv"))
    
    result_to_export = sim.result_to_export
    result_to_export.to_csv(os.path.join(out_path, "job_breakdown.csv"))

    tasks_logging_times = sim.tasks_logging_times
    tasks_logging_times.to_csv(os.path.join(out_path, "loadDelay_" + str(
        LOAD_INFORMATION_STALENESS) + "_placementDelay_" + str(PLACEMENT_INFORMATION_STALENESS) + ".csv"))
    
    sim.batch_exec_log.to_csv(os.path.join(out_path, "batch_log.csv"))
    
    worker_model_histories = pd.concat(list(map(lambda w: w.model_history_log, sim.workers)), 
                                    keys=list(map(lambda w: w.worker_id, sim.workers)), 
                                    names=['worker_id']).reset_index(level='worker_id')
    worker_model_histories = worker_model_histories.sort_values(by="start_time")
    worker_model_histories.to_csv(os.path.join(out_path, "model_history_log.csv"))

    if scheduler_type == SHEPHERD:
        ShepherdState.task_drop_log.to_csv(os.path.join(out_path, "drop_log.csv"))


if __name__ == "__main__":
    # 0. Set experiment parameters from input arguments
    if len(sys.argv) < 2:
        print("Usage: python3 run_experiments.py <experiment_scheduler0> <experiment_scheduler1> ..(centralheft|decentralheft|hashtask)")
        exit()

    scheduler_type = SCHEDULER_NAMES.inv[sys.argv[1]]
    run_experiment(scheduler_type)