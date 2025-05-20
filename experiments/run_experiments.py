import os
import sys

from schedulers.centralized.simulation_central import *
from schedulers.decentralized.simulation_decentral import *


sys.dont_write_bytecode = True




# experiment_schedulers options: centralheft | decentralheft | hashtask
experiment_schedulers = []
plotting_job_type_list = [0, 1, 2, 3]
# plotting_job_type_list = [2,3]
np.random.seed(42)

produce_breakdown = True


if __name__ == "__main__":
    # 0. Set experiment parameters from input arguments
    if len(sys.argv) < 2:
        print("Usage: python3 run_experiments.py <experiment_scheduler0> <experiment_scheduler1> ..(centralheft|decentralheft|hashtask)")
        exit()
        
    for arg in sys.argv[1:]:
        if arg == "centralheft":
            experiment_schedulers.append("centralheft")
        elif arg == "decentralheft":
            experiment_schedulers.append("decentralheft")
        elif arg == "hashtask":
            experiment_schedulers.append("hashtask")

    OUTPUT_FILE_NAMES = {}
    # 1. create folder to store the experimentation data
    current_directory = os.getcwd()
    for experiment_scheme in experiment_schedulers:
        folder_name = os.path.join(current_directory, "results/" + "/" + experiment_scheme + "/")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print("Directory created:", folder_name)
        else:
            print("Directory already exists:", folder_name)
        OUTPUT_FILE_NAMES[experiment_scheme] = folder_name
    
    # 2. Run and collect data
    if "centralheft" in experiment_schedulers:
        sim = Simulation_central(simulation_name="centralheft", job_split="PER_TASK",
                                    num_workers=TOTAL_NUM_OF_WORKERS, job_types_list=plotting_job_type_list, 
                                    produce_breakdown=True)
        sim.run()

        event_log = sim.event_log
        event_log.to_csv(OUTPUT_FILE_NAMES["centralheft"] + "events_by_time.csv")

        # result_to_export = sim.result_to_export
        tasks_logging_times = sim.tasks_logging_times
        tasks_logging_times.to_csv(OUTPUT_FILE_NAMES["centralheft"] + "loadDelay_" + str(
            LOAD_INFORMATION_STALENESS) + "_placementDelay_" + str(PLACEMENT_INFORMATION_STALENESS) + ".csv")

    if "hashtask" in experiment_schedulers:
        OUTPUT_FILENAME = "hashtask"
        sim = Simulation_central(simulation_name="hashtask", job_split="PER_TASK",
                                    num_workers=TOTAL_NUM_OF_WORKERS, job_types_list=plotting_job_type_list,
                                    produce_breakdown=True)
        sim.run()

        event_log = sim.event_log
        event_log.to_csv(OUTPUT_FILE_NAMES["hashtask"] + "events_by_time.csv")
        
        tasks_logging_times = sim.tasks_logging_times
        tasks_logging_times.to_csv(OUTPUT_FILE_NAMES["hashtask"] + "loadDelay_" + str(
            LOAD_INFORMATION_STALENESS) + "_placementDelay_" + str(PLACEMENT_INFORMATION_STALENESS) + ".csv")

    if "decentralheft" in experiment_schedulers:
        OUTPUT_FILENAME = "decentralheft"

        sim = Simulation_decentral(simulation_name="decentralheft", job_split="PER_TASK",
                                    num_workers=TOTAL_NUM_OF_WORKERS, job_types_list=plotting_job_type_list,
                                    dynamic_adjust=False, \
                                    consider_load=True, \
                                    consider_cache=True, \
                                    produce_breakdown=True)
        sim.run()
        
        # dataframe = sim.result_to_export
        # dataframe.to_csv(OUTPUT_FILE_NAMES["decentralheft"] + "loadDelay_" + str(
        #     LOAD_INFORMATION_STALENESS) + "_placementDelay_" + str(PLACEMENT_INFORMATION_STALENESS) + ".csv")
        
        event_log = sim.event_log
        event_log.to_csv(OUTPUT_FILE_NAMES["decentralheft"] + "events_by_time.csv")
        
        result_to_export = sim.result_to_export
        result_to_export.to_csv(OUTPUT_FILE_NAMES["decentralheft"] + "job_breakdown.csv")

        tasks_logging_times = sim.tasks_logging_times
        tasks_logging_times.to_csv(OUTPUT_FILE_NAMES["decentralheft"] + "loadDelay_" + str(
            LOAD_INFORMATION_STALENESS) + "_placementDelay_" + str(PLACEMENT_INFORMATION_STALENESS) + ".csv")
