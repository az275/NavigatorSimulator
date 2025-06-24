import queue
from collections import defaultdict
from workers.worker import *
from core.network import *
from core.events import *
from schedulers.algo.nav_heft_algo import *


class TaskWorker(Worker):
    def __init__(self, simulation, worker_id, total_memory):
        super().__init__(simulation, worker_id, total_memory)
        # {task_obj1:[(preq_task_id0,arrival_time0), (preq_taks_id0, arrival_time1), ...], task2:[( ...],}
        self.waiting_tasks_buffer = defaultdict(lambda: [])
        # keep track of the queue information at time:  [ (time1,[task0,task1,]), (time2,[task1,...]),...]
        self.queue_history = {}
        self.involved = False
        self.max_wait_times = {}

    def add_task(self, current_time, task):
        """
        Add task into the local task queue
        """
        if task.model != None and task.model not in self.GPU_state.placed_models(current_time):
            print("Static allocation received task that cannot be executed")
            print(f"Allocated: {self.GPU_state.state_at(current_time)}, Requested ID: {task.model.model_id}")
            assert(False)

        # Update when the task is sent to the worker
        assert (task.log.task_placed_on_worker_queue_timestamp <= current_time)
        self.add_task_to_queue_history(task, current_time) # Update when the task is sent to the worker

        # Initialize max wait time
        if task.task_type not in self.max_wait_times or self.max_wait_times[task.task_type] < 0:
            self.max_wait_times[task.task_type] = current_time + task.max_wait_time
        
        return self.maybe_start_batch(current_time, task.task_type)
    
    def get_next_models(self, lookahead_count: int, current_time: float, info_staleness=0):
        if lookahead_count <= 0:
            return []
        
        next_models = []
        task_types_by_arrival, task_queues = self.get_sorted_task_types(current_time)
        for task_type in task_types_by_arrival:
            next_model = task_queues[task_type][0].model
            if next_model != None and next_model not in next_models:
                next_models.append(next_model)
            if len(next_models) == lookahead_count:
                return next_models

        return next_models

    def free_slot(self, current_time, model, task_type):
        """ Attempts to launch another task. """
        if model != None:
            self.GPU_state.release_busy_copy(model, current_time)

        get_task_events = []
        for task_type in self.queue_history.keys():
            batch_end_events = self.maybe_start_batch(current_time, task_type)
            get_task_events += batch_end_events
        
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
        initial_tasks = [task for task in job.tasks if len(task.required_task_ids) == 0]
        for initial_task in initial_tasks:
            worker_index = activation_graph[initial_task.task_id]
            task_arrival_time = current_time
            if(worker_index != self.worker_id):
                task_arrival_time = current_time + \
                    CPU_to_CPU_delay(initial_task.input_size)
            task_arrival_events.append(EventOrders(
                task_arrival_time, TaskArrival(self.simulation.workers[worker_index], initial_task, job.id)))
        
        return task_arrival_events

    #  ---------------------------  TASK EXECUTION  ----------------------

    _CAN_RUN_NOW = 0
    _CAN_RUN_ON_EVICT = 1
    _CANNOT_RUN = 2

    def can_run_task(self, current_time: float, model: Model, info_staleness=0) -> int:
        """
            Returns _CAN_RUN_NOW if model None, or model is on GPU and not currently in use.
            Returns _CAN_RUN_ON_EVICT if model can be loaded onto the GPU upon evicting
            unused models.
            Returns _CANNOT_RUN otherwise.
        """
        if model == None or self.GPU_state.does_have_idle_copy(model, current_time):
            return self._CAN_RUN_NOW
        
        # cannot load additional copies of the same model
        if any(map(lambda s: s.model == model, self.GPU_state.state_at(current_time))):
            return self._CANNOT_RUN
        
        if self.GPU_state.can_fetch_model(model, current_time):
            return self._CAN_RUN_NOW
        
        if self.GPU_state.can_fetch_model_on_eviction(model, current_time):
            return self._CAN_RUN_ON_EVICT
        
        return self._CANNOT_RUN
    
    def _maybe_start_batch(self, task_queue: list[Task], current_time: float) -> list[EventOrders]:
        """
            Attempts to start a batch drawn from [task_queue]. If there is not
            enough GPU memory or the [task_queue] is empty, does nothing. If a
            batch is started, updates task type's next wake up to max_wait_time +
            earliest remaining task's arrival.
        """
        # only wake up if existing tasks to avoid congestion since
        # empty queue will wake up on next task enqueue
        if len(task_queue) == 0:
            return []

        batch = []
        batch_end_events = []

        can_run = self.can_run_task(current_time, task_queue[0].model)
        if can_run == self._CAN_RUN_ON_EVICT:
            current_time += self.evict_models_from_GPU_until(
                current_time, task_queue[0].model.model_size, self.LOOKAHEAD_EVICTION)
        
        if can_run == self._CAN_RUN_NOW or can_run == self._CAN_RUN_ON_EVICT:
            queued_tasks = queue.Queue()
            [queued_tasks.put(task) for task in task_queue]

            # form largest batch < max_batch_size possible
            while (not queued_tasks.empty()) and len(batch) < task_queue[0].max_batch_size:
                task = queued_tasks.get()
                if (current_time >= task.log.task_placed_on_worker_queue_timestamp):
                    batch.append(task)
            
            if len(batch) > 0:
                batch_end_events, task_end_time = self.batch_execute(batch, current_time)
                for task in batch: # rm all tasks in batch
                    self.rm_task_in_queue_history(task, current_time)

                # if successfully launched batch, reset max wait time
                if not queued_tasks.empty():
                    earliest_remaining_arrival = -1
                    while not queued_tasks.empty():
                        task = queued_tasks.get()
                        if earliest_remaining_arrival < 0 or \
                            task.log.task_placed_on_worker_queue_timestamp < earliest_remaining_arrival:
                            earliest_remaining_arrival = task.log.task_placed_on_worker_queue_timestamp
                    self.max_wait_times[batch[0].task_type] = earliest_remaining_arrival + batch[0].max_wait_time
                else:
                    self.max_wait_times[batch[0].task_type] = -1

        return batch_end_events
    
    def maybe_start_batch(self, current_time: float, task_type: tuple[int, int]):
        """
            Attempts to launch a batch of [task_type]. Does nothing if there are no
            tasks of [task_type] queued.
        """
        task_queue = self.get_queue_history(current_time, task_type, info_staleness=0)
        return self._maybe_start_batch(task_queue, current_time)

    def batch_execute(self, tasks, current_time):
        """
            Fetches a new copy or reserves an idle copy of any required GPU models
            and executes the batch [tasks]. Returns a list containing the 
            BatchEndEvent and the batch execution end time.
        """
        assert(len(tasks) > 0) # cannot launch empty batch

        self.involved = True

        for task in tasks:
            task.executing_worker_id = self.worker_id

        batch_exec_time = tasks[0].task_exec_duration if tasks[0].model is None else \
            tasks[0].get_batch_exec_time(len(tasks), self.total_memory)
        
        model_fetch_time = 0
        if tasks[0].model != None:
            if self.GPU_state.does_have_idle_copy(tasks[0].model, current_time):
                self.GPU_state.reserve_idle_copy(tasks[0].model, current_time, current_time+batch_exec_time)
            else:
                model_fetch_time = self.fetch_model(tasks[0].model, current_time, exec_time=batch_exec_time)

        
        task_end_time = current_time + model_fetch_time + batch_exec_time
        task_end_events = []

        job_ids = [] # for logging

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

        self.simulation.batch_exec_log.loc[len(self.simulation.batch_exec_log)] = {
            "time": current_time,
            "worker_id": self.worker_id,
            "workflow_id": tasks[0].task_type[0],
            "task_id": tasks[0].task_id,
            "batch_size": len(tasks),
            "model_exec_time": batch_exec_time,
            "batch_exec_time": model_fetch_time + batch_exec_time,
            "job_ids": job_ids
        }

        task_end_events.append(EventOrders(current_time, BatchStartEvent(
            self, tasks[0].model, job_ids=job_ids, task_type=tasks[0].task_type
        )))
        task_end_events.append(EventOrders(task_end_time, BatchEndEvent(
            self, tasks[0].model, job_ids=job_ids, task_type=tasks[0].task_type
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
        # 0. Base case (first entry)
        if task.task_type not in self.queue_history:
            self.queue_history[task.task_type] = [(current_time, [task])]
            return

        # 1. Find the time_stamp place to add this queue information
        last_index = len(self.queue_history[task.task_type]) - 1
        while last_index >= 0:
            if self.queue_history[task.task_type][last_index][0] == current_time:
                if task not in self.queue_history[task.task_type][last_index][1]:
                    self.queue_history[task.task_type][last_index][1].append(task)
                break
            if self.queue_history[task.task_type][last_index][0] < current_time:
                # print("2")
                if task not in self.queue_history[task.task_type][last_index][1]:
                    next_queue = self.queue_history[task.task_type][last_index][1].copy()
                    next_queue.append(task)
                    last_index += 1
                    self.queue_history[task.task_type].insert(
                        last_index, (current_time, next_queue)
                    )
                break
            # check the previous entry
            last_index -= 1

        # 2. added the task to all the subsequent timestamp tuples
        while last_index < len(self.queue_history[task.task_type]):
            if task not in self.queue_history[task.task_type][last_index][1]:
                self.queue_history[task.task_type][last_index][1].append(task)
            last_index += 1

    def rm_task_in_queue_history(self, task, current_time):
        # 0. base case: shouldn't happen
        if task.task_type not in self.queue_history:
            AssertionError("rm model cached location to an empty list")
            return

        last_index = len(self.queue_history[task.task_type]) - 1
        
        # 1. find the place to add this remove_event to the tuple list
        while last_index >= 0:
            if self.queue_history[task.task_type][last_index][0] == current_time:
                if task in self.queue_history[task.task_type][last_index][1]:
                    self.queue_history[task.task_type][last_index][1].remove(task)
                break
            if self.queue_history[task.task_type][last_index][0] < current_time:
                if task in self.queue_history[task.task_type][last_index][1]:
                    next_tasks_in_queue = self.queue_history[task.task_type][last_index][1].copy()
                    next_tasks_in_queue.remove(task)
                    last_index = last_index + 1
                    self.queue_history[task.task_type].insert(
                        last_index, (current_time, next_tasks_in_queue)
                    )
                break
            last_index -= 1  # go to prev time
        # 2. remove the task from all the subsequent tuple
        while last_index < len(self.queue_history[task.task_type]):
            if task in self.queue_history[task.task_type][last_index]:
                self.queue_history[task.task_type][last_index][1].remove(task)
            last_index += 1  # do this for the remaining element after

    def get_queue_history(self, current_time, task_type, info_staleness=0) -> list:
        return self.get_history(self.queue_history[task_type], current_time, info_staleness)

    def get_task_queue_waittime(self, current_time, task_type, info_staleness=0, requiring_worker_id=None):
        if requiring_worker_id != None and requiring_worker_id != self.worker_id:
            info_staleness = 0

        task_model_id = WORKFLOW_LIST[task_type[0]]["TASKS"][task_type[1]]["MODEL_ID"]
        if task_model_id < 0:
            return 0
        
        task_model_states = list(filter(lambda s: s.model.model_id == task_model_id, 
                                        self.GPU_state.placed_model_states(current_time)))
        if len(task_model_states) == 0:
            return np.inf

        if self.GPU_state.does_have_idle_copy(task_model_states[0].model, current_time):
            return 0
        
        return min(s.reserved_until for s in task_model_states) - current_time
