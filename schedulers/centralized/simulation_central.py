from queue import PriorityQueue
import numpy as np

from core.network import CPU_to_CPU_delay
from core.simulation import *
from core.config import *
from schedulers.algo.nav_heft_algo import *
from schedulers.algo.flex_algo import *
from schedulers.algo.herd_algo import *
from workers.taskworker import *
from workers.jobworker import *
from schedulers.centralized.shepherd.shepherd_state import ShepherdState


class Simulation_central(Simulation):
    
    def __init__(self, simulation_name="", job_split="", num_workers=1, job_types_list=[0], produce_breakdown=False):

        Simulation.__init__(self, simulation_name=simulation_name, job_split=job_split,\
                            centralized_scheduler=True,\
                            total_workers=num_workers,\
                            job_types_list=job_types_list,\
                            produce_breakdown=produce_breakdown)
        self.remaining_jobs = TOTAL_NUM_OF_JOBS
        self.event_queue = PriorityQueue()

        self.model_queues = {}      # model id -> list[Task]

        self.initialize_workers()     

    def initialize_workers(self):
        # TODO: shepherd model initialization/preloading
        if self.simulation_name == "shepherd":
            self.workers = []

            task_types = get_task_types(self.job_types_list)
            models_by_wf = list(self.metadata_service.job_type_models.values())
            all_models = [m for jt in self.job_types_list for m in models_by_wf[jt]]
            task_tputs = {(0,0): 270, (0,1): 45, (0,2): 270, (0,3): 70}
            (group_sizes, stream_groups) = get_herd_assignment(task_types, all_models, task_tputs)
            
            worker_groups = []
            worker_counter = 0
            for i, group_size in enumerate(group_sizes):
                group_workers = [ShepherdWorker(self, worker_counter+j, 24, i) for j in range(int(group_size))]
                worker_counter += len(group_workers)
                self.workers += group_workers
                worker_groups.append(group_workers)

            task_type_assignments = {}
            for (sid, group_id) in stream_groups:
                task_type_assignments[task_types[sid]] = group_id

            self.state = ShepherdState(worker_groups, task_type_assignments)

            for worker in self.workers:
                # randomly choose a model to prefetch
                group_model_ids = self.state.group_models[worker.group_id]
                preloaded_model_id = np.random.choice(list(group_model_ids))
                preloaded_model = [m for m in models_by_wf[0] if m.model_id == preloaded_model_id][0]
                worker.GPU_state.prefetch_model(preloaded_model)

            self.initialize_external_clients()
        else:
            super().initialize_workers()

    def schedule_job_and_send_tasks(self, job, current_time):
        if(self.simulation_name == "centralheft"):
            return self.nav_heft_schedule_job_and_send_tasks(job,  current_time)
        elif self.simulation_name == "shepherd":
            return flex_schedule_job_on_arrival(self, self.state, self.model_queues, job, current_time)
        elif(self.simulation_name == "hashtask"):
            return self.hash_schedule_job_and_send_tasks(job, current_time)
        
    def schedule_tasks_on_arrival(self, tasks, current_time):
        assert(self.simulation_name == "shepherd")
        arrived_groups = set()
        for task in tasks:
            if task.model.model_id not in self.model_queues:
                self.model_queues[task.model.model_id] = PriorityQueue()
            self.model_queues[task.model.model_id].put(OrderedTask(task))
            arrived_groups.add(self.state.task_type_to_group[task.task_type])
        events = []
        for group in arrived_groups:
            events += flex_schedule_tasks_on_arrival(
                self, self.state, group, self.model_queues, current_time)
        return events

    def add_job_completion_time(self, job_id, task_id, completion_time):
        job_is_completed = self.jobs[job_id].job_completed(
            completion_time, task_id)
        if job_is_completed:
            self.remaining_jobs -= 1

    def run(self):
        job_create_interval = DEFAULT_CREATION_INTERVAL_PERCLIENT / len(self.external_clients)
        for external_client_id in range(len(self.external_clients)):
            self.event_queue.put(EventOrders(
                external_client_id * job_create_interval, \
                JobCreationAtExternalClient(self, external_client_id)))

        last_time = 0
        while self.remaining_jobs > 0:
            cur_event = self.event_queue.get()
            
            if type(cur_event.event) in [BatchStartEvent] and \
                cur_event.event.worker.did_abandon_batch(cur_event.event.batch_id):
                continue

            if type(cur_event.event) in [BatchEndEvent] and \
                cur_event.event.worker.did_abandon_batch(cur_event.event.batch.id):
                continue
            
            print(cur_event.to_string())
            print(f"Jobs left: {self.remaining_jobs}")

            worker_id = -1
            if type(cur_event.event) == JobArrivalAtWorker:
                worker_id = cur_event.event.worker_id
            elif type(cur_event.event) not in [JobCreationAtExternalClient, JobArrivalAtScheduler, TasksArrivalAtScheduler]:
                worker_id = cur_event.event.worker.worker_id

            assert type(cur_event.event) != TaskArrival

            self.event_log.loc[len(self.event_log)] = [cur_event.current_time, worker_id, cur_event.event.to_string()]

            assert cur_event.current_time >= last_time
            last_time = cur_event.current_time
            new_events = cur_event.event.run(cur_event.current_time)
            for new_event in new_events:
                last_time = cur_event.current_time
                self.event_queue.put(new_event)
        self.run_finish(last_time, by_job_type=True)
        

    def nav_heft_schedule_job_and_send_tasks(self, job,  current_time):
        """ HEFT scheduler to schedule Tasks and send the initial task to worker """
        task_arrival_events = []  # List to store the TaskArrivalEvent to the receiving Workers

        # 1. compute scheduling decisions
        # {task_id0->worker_id0, ...}
        activation_graph = nav_heft_job_plan(
            job, self.workers, current_time)
        # 2. assign the planned ADFG to job object
        job.assign_ADFG(activation_graph)

        # 3. send the first task to allocated worker
        initial_task = job.tasks[0]
        task_arrival_time = current_time + \
            CPU_to_CPU_delay(initial_task.input_size)
        worker_index = activation_graph[initial_task.task_id]
        task_arrival_events.append(EventOrders(
            task_arrival_time, TaskArrival(self.workers[worker_index], initial_task, job.id)))
        return task_arrival_events

    def hash_schedule_job_and_send_tasks(self, job, current_time):
        """ hash scheme to execute Tasks and then send the initial task to worker """
        task_arrival_events = []  # List to store the TaskArrivalEvent to the receiving Workers

        # 1. assign the task in job object to the worker based on hashing
        activation_graph = {}  # {task_id0->worker_id0, ...}
        for task in job.tasks:
            allocated_worker_id = np.random.choice(
                range(self.total_workers), replace=True)
            activation_graph[task.task_id] = allocated_worker_id
        job.assign_ADFG(activation_graph)

        # 2. send the first task to allocated worker
        initial_task = job.tasks[0]
        task_arrival_time = current_time + \
            CPU_to_CPU_delay(initial_task.input_size)
        worker_index = activation_graph[initial_task.task_id]
        task_arrival_events.append(EventOrders(
            task_arrival_time, TaskArrival(self.workers[worker_index], initial_task, job.id)))
        return task_arrival_events

    def affinity_schedule_job_and_send_tasks(self, job, current_time):
        """ hash scheme to execute Tasks and then send the initial task to worker """
        task_arrival_events = []  # List to store the TaskArrivalEvent to the receiving Workers

        # 1. assign the task in job object to the worker based on hashing

        # task_arrival_events.append(EventOrders(
        #     task_arrival_time, TaskArrival(self.workers[worker_index], initial_task, job.id)))
        return task_arrival_events
