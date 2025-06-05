'''
This simulation experiment framework of event generation 
is referenced from Sparrow: https://github.com/radlab/sparrow 
'''

import numpy as np
from matplotlib import pyplot as plt
from core.config import *
from core.metadata_service import *
from core.print_utils import *
from core.external_client import *
from core.events import *
import pandas as pd


class Simulation(object):
    def __init__(
            self,
            simulation_name,
            job_split,  
            centralized_scheduler=False,  
            dynamic_adjust=False,  
            total_workers=1,
            slots_per_worker=1,
            job_types_list=[0],
            produce_breakdown=False,
            use_boost=False,
    ):
        self.simulation_name = simulation_name
        self.centralized_scheduler = centralized_scheduler
        self.job_split = job_split
        self.total_workers = total_workers
        self.slots_per_worker = slots_per_worker
        self.job_types_list = job_types_list
        self.job_split = job_split
        self.dynamic_adjust = dynamic_adjust
        self.workers = []
        self.metadata_service = MetadataService()
        self.external_clients = []
        self.use_boost = use_boost

        JobCreationAtExternalClient.job_creation_counter = 0
        self.jobs = {}
        
        # Tracking measurements
        self.result_to_export = pd.DataFrame()
        self.tasks_logging_times = pd.DataFrame()
        self.event_log = pd.DataFrame(columns=["time", "event"])

        print("---- SIMULATION : " + self.simulation_name + "----")
        self.produce_breakdown =  produce_breakdown

    def initialize_model_placement_at_workers(self):
        """Initial object placement to home node"""
        # Hashing Scheme: associate every Model to a home Node (randomly)
        all_models = list(self.metadata_service.job_type_models.values())
        flattened_all_models = [
            model for sublist_models in all_models for model in sublist_models]
        rand_worker_indices = np.random.choice(range(self.total_workers),
                                               size=len(flattened_all_models), replace=True)
        for worker_index, model in zip(rand_worker_indices, flattened_all_models):
            self.workers[worker_index].initial_model_placement(model)

    def initialize_external_clients(self):
        for job_type_id in self.job_types_list:
            self.external_clients.append(
                ExternalClient(self, job_type=job_type_id))
    """
     --------------    Printing out the simulation results     --------------
    """


    def run_finish(self, last_time, by_job_type=False):
        # 1. Get the completed job list to compute statistics 
        completed_jobs = [j for j in self.jobs.values() if len(
            j.completed_tasks) == len(j.tasks)]
        print_end_jobs(last_time, completed_jobs, self.jobs)
        completed_jobs = completed_jobs[int(len(completed_jobs) / 10):] # ignore the warnup jobs
        # 2. Compute the metrics of interest
        response_times = [job.end_time -
                               job.create_time for job in completed_jobs]
        slow_down_rate = [(job.end_time - job.create_time) /
                               WORKFLOW_LIST[job.job_type_id]["BEST_EXEC_TIME"] for job in completed_jobs]
        print_response_time(response_times)
        print_slowdown(slow_down_rate)
        ADFG_created = []
        for job in completed_jobs:
            if job.ADFG not in ADFG_created:
                ADFG_created.append(job.ADFG)
                # print(job.job_type_id , job.ADFG)
        print(".... number of DAG created: {}".format(len(ADFG_created)))
        print_involved_workers(self.workers)
        if by_job_type:
            response_time_per_type = {}
            slow_down_per_type = {}
            for job in completed_jobs:
                if job.job_type_id not in response_time_per_type:
                    response_time_per_type[job.job_type_id] = []
                    slow_down_per_type[job.job_type_id] = []
                response_time_per_type[job.job_type_id].append(job.end_time - job.create_time)
                slow_down_per_type[job.job_type_id].append((job.end_time - job.create_time) / WORKFLOW_LIST[job.job_type_id]["BEST_EXEC_TIME"])
            # print statistics for each job type
            print_stats_by_job_type(response_time_per_type, slow_down_per_type)
        if self.produce_breakdown:
            self.produce_time_breakdown_results(completed_jobs)

    def produce_time_breakdown_results(self, completed_jobs):

        dataframe = pd.DataFrame(columns=["job_id", "load_info_staleness", "placement_info_staleness", "req_inter_arrival_delay",
                                          "workflow_type", "job_create_time", "scheduler_type", "slowdown", "response_time"])
        dataframe_tasks_log = pd.DataFrame(columns=["workflow_type", "task_id", "task_arrival_time", "task_start_exec_time", "time_to_buffer", "dependency_wait_time",
                                                    "time_spent_in_queue", "model_fetching_time", "execution_time"])

        for index, completed_job in enumerate(completed_jobs):

            if index < 0:  # ignore the first 50 jobs
                continue

            slowdown = (completed_job.end_time - completed_job.create_time) / \
                WORKFLOW_LIST[completed_job.job_type_id]["BEST_EXEC_TIME"]
            response_time = completed_job.end_time - completed_job.create_time
            job_creation_interval = DEFAULT_CREATION_INTERVAL_PERCLIENT
            if "JOB_CREATION_INTERVAL" in WORKFLOW_LIST[completed_job.job_type_id]:
                job_creation_interval = WORKFLOW_LIST[completed_job.job_type_id]["JOB_CREATION_INTERVAL"]
            dataframe.loc[index] = [index, LOAD_INFORMATION_STALENESS, PLACEMENT_INFORMATION_STALENESS, job_creation_interval, completed_job.job_type_id,
                                    completed_job.create_time, self.simulation_name, slowdown, response_time]

        task_index = 0
        for job in completed_jobs:
            for task in job.tasks:
                time_to_buffer = task.log.task_arrival_at_worker_buffer_timestamp - \
                    task.log.job_creation_timestamp
                dependency_wait_time = task.log.task_placed_on_worker_queue_timestamp - \
                    task.log.task_arrival_at_worker_buffer_timestamp
                time_spent_in_queue = task.log.task_front_queue_timestamp - \
                    task.log.task_placed_on_worker_queue_timestamp
                model_fetching_time = task.log.get_model_fetch_time()
                execution_time = task.log.task_execution_end_timestamp - \
                    task.log.task_execution_start_timestamp

                assert time_to_buffer >= 0
                assert dependency_wait_time >= 0
                assert time_spent_in_queue >= 0
                assert model_fetching_time >= 0
                assert execution_time >= 0

                dataframe_tasks_log.loc[task_index] = [job.job_type_id, task.task_id, task.log.task_arrival_at_worker_buffer_timestamp, 
                                                       task.log.task_execution_start_timestamp,time_to_buffer, dependency_wait_time, 
                                                       time_spent_in_queue, model_fetching_time, execution_time]
                task_index += 1

        self.tasks_logging_times = dataframe_tasks_log
        self.result_to_export = dataframe
