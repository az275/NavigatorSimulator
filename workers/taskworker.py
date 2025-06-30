from workers.worker import *
from core.network import *
from core.events import *
from schedulers.algo.nav_heft_algo import *


class TaskWorker(Worker):
    def __init__(self, simulation, worker_id, total_memory):
        super().__init__(simulation, worker_id, total_memory)
        self.involved = False

    def add_task(self, current_time, task):
        """
        Add task into the local task queue
        """
        raise NotImplementedError()
    
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

    def free_slot(self, current_time, batch: Batch, task_type):
        """ Attempts to launch another task. """
        assert(not self.did_abandon_batch(batch.id))

        # finished batch clean up & logging
        if batch.model != None:
            self.GPU_state.release_busy_model(batch.id, current_time)

        for task in batch.tasks:
            self.simulation.add_job_completion_time(
                task.job_id, task.task_id, current_time)
            
            # task log tracking
            task.log.task_front_queue_timestamp = batch.front_queue_timestamp
            task.log.task_execution_start_timestamp = batch.execution_start_timestamp
            task.log.task_execution_end_timestamp = current_time

        self.simulation.batch_exec_log.loc[len(self.simulation.batch_exec_log)] = {
            "start_time": batch.execution_start_timestamp,
            "end_time": current_time,
            "worker_id": self.worker_id,
            "workflow_id": batch.tasks[0].task_type[0],
            "task_id": batch.tasks[0].task_id,
            "batch_size": batch.size(),
            "job_ids": batch.job_ids
        }

        events = self.send_results_to_next_workers(current_time, batch)

        return events

    #  ---------------------------  TASK EXECUTION  ----------------------

    def maybe_start_batch(self, batch: Batch, current_time: float) -> list[EventOrders]:
        """
            Attempts to start [batch], given enough GPU memory. If a
            batch is started, updates task type's next wake up to max_wait_time +
            earliest remaining task's arrival.
        """
        batch_end_events = []

        can_run = self.can_run_task(current_time, batch.model)
        if can_run == self._CAN_RUN_ON_EVICT:
            current_time += self.evict_models_from_GPU_until(
                current_time, batch.model.model_size, self.LOOKAHEAD_EVICTION)
        
        if can_run == self._CAN_RUN_NOW or can_run == self._CAN_RUN_ON_EVICT:
            batch_end_events, task_end_time = self.batch_execute(batch, current_time)
        
        return batch_end_events

    def batch_execute(self, batch: Batch, current_time: float):
        """
            Fetches a new copy or reserves an idle copy of any required GPU models
            and executes the batch [tasks]. Returns a list containing the 
            BatchEndEvent and the batch execution end time.
        """
        self.involved = True

        for task in batch.tasks:
            task.executing_worker_id = self.worker_id

        batch_exec_time = batch.tasks[0].task_exec_duration if batch.model is None else \
            batch.tasks[0].get_batch_exec_time(batch.size(), self.total_memory)
        
        model_fetch_time = 0
        if batch.model != None:
            if self.GPU_state.does_have_idle_copy(batch.model, current_time):
                self.GPU_state.reserve_idle_copy(
                    batch.model, current_time, batch, current_time+batch_exec_time)
            else:
                assert(False)
                model_fetch_time = self.fetch_model(
                    batch.model, current_time, batch, exec_time=batch_exec_time)
        
        batch.front_queue_timestamp = current_time
        batch.execution_start_timestamp = current_time + model_fetch_time

        task_end_time = current_time + model_fetch_time + batch_exec_time
        task_end_events = []

        task_end_events.append(EventOrders(current_time, BatchStartEvent(
            self, batch_id=batch.id, job_ids=batch.job_ids, task_type=batch.tasks[0].task_type)))
        task_end_events.append(EventOrders(task_end_time, BatchEndEvent(
            self, batch, job_ids=batch.job_ids, task_type=batch.tasks[0].task_type)))
        return task_end_events, task_end_time

    #  ---------------------------  Subsequent TASK Transfer   --------------------

    def send_result_to_next_workers(self, current_time, task) -> list:
        """
        Send the result of a task to the next worker in the inference pipeline (it may be the same worker)
        """
        raise NotImplementedError()

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
