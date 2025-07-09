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
from workers.heft_task_worker import *
from workers.shepherd_task_worker import *

# import gurobipy as gp
# from gurobipy import GRB


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
            produce_breakdown=False
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

        self._batch_counter = 0

        self.jobs = {}
        
        # Tracking measurements
        self.result_to_export = pd.DataFrame()
        self.tasks_logging_times = pd.DataFrame()
        self.event_log = pd.DataFrame(columns=["time", "worker_id", "event"])
        self.batch_exec_log = pd.DataFrame(columns=["start_time", "end_time", "worker_id", "workflow_id", 
                                                    "task_id", "batch_size", "job_ids"])

        print("---- SIMULATION : " + self.simulation_name + "----")
        self.produce_breakdown =  produce_breakdown

    def initialize_model_placement_at_workers(self):
        """Initial object placement to home node"""
        all_models = list(self.metadata_service.job_type_models.values())
        
        models = ["0,2", "1", "3"] # index in all_models[0]
        configs = [6, 12, 24]
        nodes = [i for i in range(TOTAL_NUM_OF_NODES)]

        # ppl1
        throughput = {
            "0,2": {6: 200, 12: 240, 24: 270},
            "1": {24: 45},
            "3": {6: 55, 12: 55, 24: 70},
        }

        # ppl2
        throughput = {
            '0': {12:71, 24: 125},
            '1': {6: 5333, 12: 6083 ,24: 7555},
            '2': {6: 26, 12: 45, 24: 92},
            '3': {12: 3.9, 24: 4.82}
        }

        valid_layouts = [[24], [12, 12], [12, 6, 6], [6, 6, 6, 6]]

        # def solve_leximin(locked_lower_bounds):
        #     m = gp.Model("Leximin_Level")
        #     x, y = {}, {}
        #     T_m = {}
        #     Z = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z")

        #     for model in models:
        #         for node in nodes:
        #             for c in configs:
        #                 if c in throughput[model]:
        #                     x[model, node, c] = m.addVar(vtype=GRB.INTEGER, name=f"x_{model}_{node}_{c}")

        #     for node in nodes:
        #         for lid, layout in enumerate(valid_layouts):
        #             y[node, lid] = m.addVar(vtype=GRB.BINARY, name=f"y_{node}_{lid}")

        #     for model in models:
        #         T_m[model] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"T_{model}")
        #         m.addConstr(
        #             T_m[model] == gp.quicksum(
        #                 x[model, node, c] * throughput[model][c]
        #                 for node in nodes for c in configs if (model, node, c) in x
        #             )
        #         )

        #         if model in locked_lower_bounds:
        #             m.addConstr(T_m[model] >= locked_lower_bounds[model])
        #         else:
        #             m.addConstr(Z <= T_m[model])

        #         m.addConstr(gp.quicksum(
        #             x[model, node, c] for node in nodes for c in configs if (model, node, c) in x
        #         ) >= 1)

        #     for node in nodes:
        #         m.addConstr(gp.quicksum(y[node, lid] for lid in range(len(valid_layouts))) == 1)
        #         for c in configs:
        #             m.addConstr(
        #                 gp.quicksum(
        #                     x[model, node, c] for model in models if (model, node, c) in x
        #                 ) <= gp.quicksum(
        #                     y[node, lid] * layout.count(c)
        #                     for lid, layout in enumerate(valid_layouts)
        #                 )
        #             )

        #     m.setObjective(Z, GRB.MAXIMIZE)
        #     m.setParam("OutputFlag", 0)
        #     m.optimize()

        #     throughput_vals = {model: T_m[model].X for model in models}
        #     assignment = {
        #         (model, node, c): int(x[model, node, c].X)
        #         for model in models for node in nodes for c in configs
        #         if (model, node, c) in x and x[model, node, c].X > 0.5
        #     }
        #     return throughput_vals, assignment

        # # Leximin loop
        # locked = {}
        # for _ in range(len(models)):
        #     T_vals, assignment = solve_leximin(locked)
        #     unlocked = [m for m in models if m not in locked]
        #     if not unlocked:
        #         break
        #     min_model = min(unlocked, key=lambda m: T_vals[m])
        #     locked[min_model] = T_vals[min_model]

        # static experiment alloc, ppl1:
        worker_configs = [
            (24, [all_models[0][1]]),
            (24, [all_models[0][1]]),
            (24, [all_models[0][1]]),
            (6, [all_models[0][3]]),
            (6, [all_models[0][3]]),
            (6, [all_models[0][3]]),
            (6, [all_models[0][0], all_models[0][2]])
        ]

        # static experiment alloc, ppl2:
        worker_configs += [
            (12, [all_models[1][0]]),
            (12, [all_models[1][1], all_models[1][2]]),
            (12, [all_models[1][3]]),
            (12, [all_models[1][3]]),
            (12, [all_models[1][3]]),
            (12, [all_models[1][3]]),
            (12, [all_models[1][3]]),
            (12, [all_models[1][3]])
        ]

        # static Gurobi alloc:
        # for (model_idxs, node, c), count in assignment.items():
        #     models = list(map(lambda idx: all_models[0][int(idx)], model_idxs.split(",")))
        #     for _ in range(count):
        #         worker_configs.append((c, models))
        #     print(f" - Model {model_idxs} assigned {count}x to {node} with MIG {c}GB")
        return worker_configs
    
    def initialize_workers(self):
        if self.job_split == "PER_TASK":
            worker_configs = self.initialize_model_placement_at_workers()
            for i, config in enumerate(worker_configs):
                if self.simulation_name == "shepherd":
                    self.workers.append(ShepherdWorker(self, i, config[0], 0))
                else:
                    self.workers.append(HeftTaskWorker(self, i, config[0]))
                for model in config[1]:
                    self.metadata_service.add_model_cached_location(model, i, 0)
                    self.workers[-1].GPU_state.prefetch_model(model)
            self.initialize_external_clients()

    def initialize_external_clients(self):
        for job_type_id in self.job_types_list:
            self.external_clients.append(
                ExternalClient(self, job_type=job_type_id))
    
    def send_rate_at(self, workflow: int, time: float) -> float:
        if time == 0:
            return SEND_RATES_BY_WORKFLOW[workflow]["SEND_RATES"][0]
        for i, change_time in enumerate(self.send_rate_change_times[::-1]):
            if time >= change_time:
                return SEND_RATES_BY_WORKFLOW[workflow]["SEND_RATES"][-(i+1)]
        return SEND_RATES_BY_WORKFLOW[workflow]["SEND_RATES"][0]
            
    def generate_all_jobs(self):
        self.send_rate_change_times = []

        for idx, i in enumerate(self.job_types_list):
            curr_send_rate_idx = 0
            curr_send_rate = SEND_RATES_BY_WORKFLOW[i]["SEND_RATES"][0]
            curr_time = 0

            jid_offset = sum(TOTAL_NUM_OF_JOBS_PER_WORKFLOW[x] for x in self.job_types_list[:idx])
            for j in range(TOTAL_NUM_OF_JOBS_PER_WORKFLOW[i]):
                if curr_send_rate_idx < len(SEND_RATES_BY_WORKFLOW[i]["SEND_RATES"]) - 1:
                    if j == sum(SEND_RATES_BY_WORKFLOW[i]["SEND_RATE_CHANGE_INTERVALS"][:curr_send_rate_idx+1]):
                        curr_send_rate_idx += 1
                        curr_send_rate = SEND_RATES_BY_WORKFLOW[i]["SEND_RATES"][curr_send_rate_idx]
                        self.send_rate_change_times.append(curr_time)

                next_job = self.external_clients[idx].create_job(curr_time, j + jid_offset, curr_send_rate)
                self.jobs[next_job.id] = next_job
                curr_time = next_job.create_time + CPU_to_CPU_delay(next_job.tasks[0].input_size)

                if self.centralized_scheduler:
                    self.event_queue.put(EventOrders(
                        curr_time, JobArrivalAtScheduler(self, next_job)))
                else:
                    initial_worker_id = self.external_clients[idx].select_initial_worker_id()
                    self.event_queue.put(EventOrders(
                        curr_time, JobArrivalAtWorker(self, next_job, initial_worker_id)))
    
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
        dataframe_tasks_log = pd.DataFrame(columns=["workflow_type", "task_id", "worker_id", "task_arrival_time", "task_start_exec_time", "time_to_buffer", "dependency_wait_time",
                                                    "time_spent_in_queue", "model_fetching_time", "execution_time"])

        for index, completed_job in enumerate(completed_jobs):

            if index < 0:  # ignore the first 50 jobs
                continue

            slowdown = (completed_job.end_time - completed_job.create_time) / \
                WORKFLOW_LIST[completed_job.job_type_id]["BEST_EXEC_TIME"]
            response_time = completed_job.end_time - completed_job.create_time
            job_creation_interval = self.send_rate_at(completed_job.job_type_id, completed_job.create_time)
            dataframe.loc[index] = [completed_job.id, LOAD_INFORMATION_STALENESS, PLACEMENT_INFORMATION_STALENESS, job_creation_interval, completed_job.job_type_id,
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

                dataframe_tasks_log.loc[task_index] = [job.job_type_id, task.task_id, task.executing_worker_id, task.log.task_arrival_at_worker_buffer_timestamp, 
                                                       task.log.task_execution_start_timestamp,time_to_buffer, dependency_wait_time, 
                                                       time_spent_in_queue, model_fetching_time, execution_time]
                task_index += 1

        self.tasks_logging_times = dataframe_tasks_log
        self.result_to_export = dataframe
