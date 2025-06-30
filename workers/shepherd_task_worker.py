import queue
from collections import defaultdict
from workers.worker import *
from workers.taskworker import *
from core.network import *
from core.events import *
from schedulers.algo.nav_heft_algo import *
from schedulers.algo.flex_algo import *


class ShepherdWorker(TaskWorker):
    def __init__(self, simulation, worker_id, total_memory):
        super().__init__(simulation, worker_id, total_memory)

    def free_slot(self, current_time, batch: Batch, task_type):
        """ Attempts to launch another task. """
        events = super().free_slot(current_time, batch, task_type)
        events += flex_schedule_on_batch_completion(self.simulation, self, batch, current_time)
        return events
    
    def preempt_batch(self, old_batch_id: int, new_batch: Batch, current_time: float):
        evicted_batch = self.evict_batch(old_batch_id, current_time)
        events, _ = self.batch_execute(new_batch, current_time)
        events.append(EventOrders(
            current_time + CPU_to_CPU_delay(evicted_batch.tasks[0].input_size * evicted_batch.size()), 
            TasksArrivalAtScheduler(self.simulation, evicted_batch.tasks)))
        return events

    #  ---------------------------  Subsequent TASK Transfer   --------------------

    def send_results_to_next_workers(self, current_time: float, batch: Batch) -> list:
        """
        Send the result of a task to the next worker in the inference pipeline (it may be the same worker)
        """
        # TODO: CPU to CPU delay is correct?
        new_tasks = []
        for task in batch.tasks:
            new_tasks += task.job.newly_available_tasks(task)
        if len(new_tasks) > 0:
            return [EventOrders(
                current_time + CPU_to_CPU_delay(task.result_size), 
                TasksArrivalAtScheduler(self.simulation, new_tasks))]
        return []