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
        # if HERD, dynamic loading must be enabled
        assert(ALLOCATION_STRATEGY != "HERD" or ENABLE_DYNAMIC_MODEL_LOADING)
        
        self.remaining_jobs = sum(TOTAL_NUM_OF_JOBS_PER_WORKFLOW[i] for i in job_types_list)
        self.event_queue = PriorityQueue()

        self.model_queues = {}      # model id -> list[Task]

        self.next_worker_id = { jt: [0 for tt in get_task_types([jt])] for jt in job_types_list }

        self.initialize_workers()

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
            self.model_queues[task.model.model_id].put(OrderedTask(task, current_time))
            arrived_groups.add(self.state.task_type_to_group[task.task_type])
        events = []
        for group in arrived_groups:
            events += flex_schedule_tasks_on_arrival(
                self, self.state, group, self.model_queues, current_time)
        return events
    
    def schedule_tasks_on_queue(self, current_time):
        """
            Schedules tasks that may be on queue without requiring
            newly arrived tasks.
        """
        events = []
        if self.simulation_name == "shepherd":
            for group in range(len(self.state.worker_groups)):
                events += flex_schedule_tasks_on_arrival(
                    self, self.state, group, self.model_queues, current_time)
        return events

    def add_job_completion_time(self, job_id, task_id, completion_time):
        job_is_completed = self.jobs[job_id].job_completed(
            completion_time, task_id)
        if job_is_completed:
            self.remaining_jobs -= 1

    def run(self):
        self.generate_all_jobs()

        last_time = 0
        while (self.remaining_jobs - len(self.task_drop_log)) > 0:
            cur_event = self.event_queue.get()

            if cur_event.event.should_abandon_event(cur_event.current_time, {}):
                continue
            
            print(cur_event.to_string())
            print(f"Jobs left: {self.remaining_jobs}")

            worker_id = -1
            if type(cur_event.event).is_worker_event():
                worker_id = cur_event.event.worker.worker_id
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
            if ENABLE_DYNAMIC_MODEL_LOADING:
                if ALLOCATION_STRATEGY == "HERD":
                    # don't choose worker that is not in the correct group
                    while task.model and task.model.model_id not in self.state.group_models[self.workers[self.next_worker_id[task.task_type[0]][task.task_id]].group_id] and \
                        self.workers[self.next_worker_id[task.task_type[0]][task.task_id]].total_memory * 10**6 < task.model.model_size:
                        self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.workers)
                else:
                    while task.model and self.workers[self.next_worker_id[task.task_type[0]][task.task_id]].total_memory * 10**6 < task.model.model_size:
                        self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.workers)
            else:
                while task.model and all(m.model_id != task.model.model_id for m in self.workers[self.next_worker_id[task.task_type[0]][task.task_id]].GPU_state.placed_models(current_time)):
                    self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.workers)
            activation_graph[task.task_id] = self.next_worker_id[task.task_type[0]][task.task_id]
            self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.workers)
        job.assign_ADFG(activation_graph)

        # 2. send the first task to allocated worker
        initial_tasks = [task for task in job.tasks if len(task.required_task_ids) == 0]
        for initial_task in initial_tasks:
            task_arrival_time = current_time + CPU_to_CPU_delay(initial_task.input_size)
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
