import queue
from collections import defaultdict
from workers.worker import *
from core.network import *
from core.events import *
from schedulers.algo.nav_heft_algo import *
import time


class TaskWorker(Worker):
    def __init__(self, simulation, num_free_slots, worker_id):
        super().__init__(simulation, num_free_slots, worker_id)
        # {task_obj1:[(preq_task_id0,arrival_time0), (preq_taks_id0, arrival_time1), ...], task2:[( ...],}
        self.waiting_tasks_buffer = defaultdict(lambda: [])
        # keep track of the queue information at time:  [ (time1,[task0,task1,]), (time2,[task1,...]),...]
        self.queue_history = []
        self.involved = False
        self.next_check_times = {}

    def add_task(self, current_time, task):
        """
        Add task into the local task queue
        """

        # print(f"[{current_time}] W{self.worker_id}: T{task.task_id} arrived")

        # Update when the task is sent to the worker
        assert (task.log.task_placed_on_worker_queue_timestamp <= current_time)
        self.add_task_to_queue_history(task, current_time)
        _, task_end_events = self.maybe_start_task_for_type(current_time, task.task_id, task.max_wait_time, False)
        return task_end_events

    def free_slot(self, current_time):
        """ Frees a slot on the worker and attempts to launch another task in that slot. """
        self.num_free_slots += 1
        get_task_events = self.maybe_start_task_any(current_time)
        return get_task_events

    #  --------------------------- DECENTRALIZED WORKER SCHEDULING  ----------------------
    def schedule_job_heft(self, current_time, job):
        """ HEFT scheduler to schedule Tasks and send the initial task to worker 
        This implementation so far, assume scheduling thread is different from the execution thread,
            where even schedule the initial task on this same worker, 
            this task needs to be waiting from the end of queue after scheduling
        """
        task_arrival_events = []  # List to store the TaskArrivalEvent to the receiving Workers
        # 1. compute scheduling decisions based on decentralized HEFT
        # {task_id0->worker_id0, ...}
        activation_graph = nav_heft_job_plan(job, \
                                             self.simulation.workers, \
                                             current_time, \
                                             initial_worker_id=self.worker_id, \
                                             consider_load=self.simulation.consider_load, \
                                             consider_cache=self.simulation.consider_cache)

        # 2. assign the planned ADFG to job object
        job.assign_ADFG(activation_graph)

        # 3. send the first task to allocated worker
        initial_task = job.tasks[0]
        worker_index = activation_graph[initial_task.task_id]
        task_arrival_time = current_time
        if(worker_index != self.worker_id):
            task_arrival_time = current_time + \
                CPU_to_CPU_delay(initial_task.input_size)
        task_arrival_events.append(EventOrders(
            task_arrival_time, TaskArrival(self.simulation.workers[worker_index], initial_task, job.id)))
        return task_arrival_events

    #  ---------------------------  TASK EXECUTION  ----------------------

    def maybe_start_task_any(self, current_time):
        task_list = self.get_queue_history(current_time, info_staleness=0)

        queued_tasks = queue.Queue()
        [queued_tasks.put(task) for task in task_list]
        while (not queued_tasks.empty()) and self.num_free_slots > 0:
            task = queued_tasks.get()
            if (current_time >= task.log.task_placed_on_worker_queue_timestamp):
                did_exec_batch, task_end_events = self.maybe_start_task_for_type(
                    current_time, task.task_id, task.max_wait_time, False
                )
                if did_exec_batch:
                    return task_end_events
                # keep checking queue until batch is executed or tasks run out

        # if no queued tasks, maybe is never called and no wake up events are
        # appended; assume in this case worker will be woken up when any new 
        # task arrives
        return []
    

    def maybe_start_task_for_type(self, current_time, task_type, task_wait_time, do_exec_batch) -> tuple[bool, list]:
        """
            Returns did_exec_batch : bool, task_end_events : list[Event]
        """
        latest_time = current_time
        did_exec_batch = False

        task_end_events = []
        task_list = [task for task in self.get_queue_history(current_time, info_staleness=0) 
                     if task.task_id == task_type]
        
        queued_tasks = queue.Queue()
        [queued_tasks.put(task) for task in task_list]

        batch = []
        while (not queued_tasks.empty()) and self.num_free_slots > 0 and len(batch) < task_list[0].max_batch_size:
            task = queued_tasks.get()
            if (current_time >= task.log.task_placed_on_worker_queue_timestamp):
                batch.append(task)
        
        # full batch or max wait time has passed
        if len(task_list) > 0 and self.num_free_slots > 0 \
            and (do_exec_batch or len(batch) >= task_list[0].max_batch_size):

            batch_end_events, task_end_time = self.batch_execute(
                batch, current_time)
            
            # rm all tasks in batch
            for task in batch:
                self.rm_task_in_queue_history(task, current_time)

            latest_time = task_end_time

            did_exec_batch = True
            task_end_events += batch_end_events

        next_check_time = latest_time + task_wait_time

        # if idle, check again in wait time
        task_end_events.append(
            EventOrders(
                next_check_time,
                WorkerWakeUpEvent(self, task_type, task_wait_time)
            )
        )
        self.next_check_times[task_type] = next_check_time

        return did_exec_batch, task_end_events

    # modify to handle a batch of tasks:
    # need to model batch execution duration
    # transfer to next step should handle a list of tasks
    def batch_execute(self, tasks, current_time):
        self.involved = True
        self.num_free_slots -= 1
        model_fetch_time = self.fetch_model(tasks[0].model, current_time)

        batch_index = 0
        for i, batch_size in enumerate(sorted(tasks[0].batch_sizes)): # assumes batch_sizes are sorted
            if len(tasks) <= batch_size:
                batch_index = i
                break

        task_end_time = current_time + model_fetch_time + tasks[0].batch_exec_time[batch_index]
        task_end_events = []

        job_ids = []

        for task in tasks:
            events = self.send_result_to_next_workers(
                task_end_time, task)
            task_end_events += events

            self.simulation.add_job_completion_time(
                task.job_id, task.task_id, task_end_time)
            
            job_ids.append(task.job_id)
        
            # task log tracking
            task.log.task_front_queue_timestamp = current_time
            task.log.task_execution_start_timestamp = current_time + model_fetch_time
            task.log.task_execution_end_timestamp = task_end_time

        task_end_events.append(EventOrders(task_end_time, BatchEndEvent(
            self, job_ids=job_ids, task_id=tasks[0].task_id
        )))

        return task_end_events, task_end_time

    #  ---------------------------  Subsequent TASK Transfer   --------------------

    def send_result_to_next_workers(self, current_time, task) -> list:
        """
        Send the result of a task to the next worker in the inference pipeline (it may be the same worker)
        """
        events = []
        cur_job = self.simulation.jobs[task.job_id]
        for cur_task_id in task.next_task_ids:
            cur_task = cur_job.tasks[cur_task_id]
            assigned_worker_id = task.ADFG[cur_task.task_id]
            if self.simulation.dynamic_adjust:
                assigned_worker_id = nav_heft_task_adjustment(cur_job, cur_task_id, \
                                                              self.simulation.workers, \
                                                              current_time, \
                                                              self.worker_id, \
                                                              assigned_worker_id)
            next_worker = self.simulation.workers[assigned_worker_id]
            transfer_delay = 0
            if assigned_worker_id != self.worker_id:  # The next worker on the pipeline is NOT the same node
                transfer_delay = GPU_to_GPU_delay(task.result_size)
            events.append(EventOrders(current_time + transfer_delay, InterResultArrival(
                worker=next_worker, prev_task=task, cur_task=cur_task)))
        return events

    def receive_intermediate_result(self, current_time, prev_task, cur_task) -> list:
        """
        event handling when taskworker receives the result of prev_task and put it to the waiting_buffer of cur_task
        @param: current_time: the time when the prev_task result arrives at this worker
        @param: prev_task: the task that has been executed and sent the result to this worker
        @param: cur_task: the task that is waiting for the result of prev_task, to be put to the waiting_buffer
        """
        events = []
        # 0. add this arrived prev_task result to the buffer of cur_task
        if prev_task != None:
            self.waiting_tasks_buffer[cur_task].append(
                [prev_task.task_id, current_time])
        # check if we have collected all the preq_tasks for current_task to be put on the task queue
        prev_arrived_list = self.waiting_tasks_buffer[cur_task]
        if(len(prev_arrived_list) != len(cur_task.required_task_ids)):
            # the cur_task hasn't received all the prerequisite task it needs to execute. SKIP this round
            return events
        # time when all the pre-requisite tasks have arrived
        receive_time = current_time
        for element in prev_arrived_list:
            receive_time = max(receive_time, element[1])
        events.append(EventOrders(
            receive_time, TaskArrival(self, cur_task, cur_task.job_id)))
        return events

    
    # ------------------------- queue history update helper functions ---------------

    def add_task_to_queue_history(self, task, current_time):
        last_index = len(self.queue_history) - 1
        # 0. base case
        if last_index == -1:
            self.queue_history.append((current_time, [task]))
            return
        # 1. Find the time_stamp place to add this queue information
        while last_index >= 0:
            if self.queue_history[last_index][0] == current_time:
                if task not in self.queue_history[last_index][1]:
                    self.queue_history[last_index][1].append(task)
                break
            if self.queue_history[last_index][0] < current_time:
                # print("2")
                if task not in self.queue_history[last_index][1]:
                    next_queue = self.queue_history[last_index][1].copy()
                    next_queue.append(task)
                    last_index += 1
                    self.queue_history.insert(
                        last_index, (current_time, next_queue)
                    )
                break
            # check the previous entry
            last_index -= 1

        # 2. added the task to all the subsequent timestamp tuples
        while last_index < len(self.queue_history):
            if task not in self.queue_history[last_index][1]:
                self.queue_history[last_index][1].append(task)
            last_index += 1

    def rm_task_in_queue_history(self, task, current_time):
        last_index = len(self.queue_history) - 1
        # 0. base case: shouldn't happen
        if last_index == -1:
            AssertionError("rm model cached location to an empty list")
            return
        # 1. find the place to add this remove_event to the tuple list
        while last_index >= 0:
            if self.queue_history[last_index][0] == current_time:
                if task in self.queue_history[last_index][1]:
                    self.queue_history[last_index][1].remove(task)
                break
            if self.queue_history[last_index][0] < current_time:
                if task in self.queue_history[last_index][1]:
                    next_tasks_in_queue = self.queue_history[last_index][1].copy()
                    next_tasks_in_queue.remove(task)
                    last_index = last_index + 1
                    self.queue_history.insert(
                        last_index, (current_time, next_tasks_in_queue)
                    )
                break
            last_index -= 1  # go to prev time
        # 2. remove the task from all the subsequent tuple
        while last_index < len(self.queue_history):
            if task in self.queue_history[last_index]:
                self.queue_history[last_index][1].remove(task)
            last_index += 1  # do this for the remaining element after

    def get_queue_history(self, current_time, info_staleness=0) -> list:
        return self.get_history(self.queue_history, current_time, info_staleness)

    def get_task_queue_waittime(self, current_time, info_staleness=0, requiring_worker_id=None):
        if requiring_worker_id != None and requiring_worker_id != self.worker_id:
            info_staleness = 0
        queueing_tasks = self.get_queue_history(current_time, info_staleness)
        waittime = 0
        for task in queueing_tasks:
            waittime += task.task_exec_duration
        return waittime
