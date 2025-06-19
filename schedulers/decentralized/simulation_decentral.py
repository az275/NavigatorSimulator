import pandas as pd

from queue import PriorityQueue

from core.simulation import *
from core.config import *
from schedulers.algo.nav_heft_algo import *
from workers.taskworker import *


class Simulation_decentral(Simulation):
    def __init__(self, simulation_name="", job_split="", num_workers=1, job_types_list=[0], dynamic_adjust=True, consider_load=True, consider_cache=True, produce_breakdown=False):

        Simulation.__init__(self, simulation_name=simulation_name, job_split=job_split, \
                            centralized_scheduler=False, \
                            dynamic_adjust=dynamic_adjust, \
                            total_workers=num_workers, \
                            job_types_list=job_types_list,\
                            produce_breakdown=produce_breakdown)

        self.remaining_jobs = TOTAL_NUM_OF_JOBS
        self.event_queue = PriorityQueue()
        
        self.consider_load, self.consider_cache = consider_load, consider_cache

        self.initialize_workers()
        self.initialize_external_clients()


    def initialize_workers(self):
        if self.job_split == "PER_TASK":
            all_models = list(self.metadata_service.job_type_models.values())[0]
            initial_models = [[all_models[1]],
                              [all_models[1]],
                              [all_models[1]],
                              [all_models[3],all_models[3],all_models[3],all_models[0],all_models[2]]]

            for i in range(self.total_workers):
                self.workers.append(TaskWorker(self, self.slots_per_worker, i))
                for model in initial_models[i]:
                    self.metadata_service.add_model_cached_location(model, i, 0)
                    self.workers[-1].add_model_to_memory_history(model, 0)


    def add_job_completion_time(self, job_id, task_id, completion_time):
        job_is_completed = self.jobs[job_id].job_completed(
            completion_time, task_id)
        if job_is_completed:
            self.remaining_jobs -= 1


    def run(self):
        client_initialize_interval = DEFAULT_CREATION_INTERVAL_PERCLIENT / len(self.external_clients)
        for external_client_id in range(len(self.external_clients)):
            self.event_queue.put(EventOrders(
                external_client_id * client_initialize_interval, \
                JobCreationAtExternalClient(self, external_client_id)))
        last_time = 0
        while self.remaining_jobs > 0:
            cur_event = self.event_queue.get()

            print(cur_event.to_string())
            print(self.remaining_jobs)
            
            self.event_log.loc[len(self.event_log)] = [cur_event.current_time, cur_event.to_string()]

            assert cur_event.current_time >= last_time
            last_time = cur_event.current_time
            new_events = cur_event.event.run(cur_event.current_time)
            for new_event in new_events:
                last_time = cur_event.current_time
                self.event_queue.put(new_event)
        self.run_finish(last_time, by_job_type=True)
        
        
