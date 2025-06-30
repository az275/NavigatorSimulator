from queue import PriorityQueue

from core.config import *
from core.task import Task
from core.job import Job
from core.batch import Batch
from core.events import *
from core.network import *

from workers.worker import Worker


class OrderedTask:
    """
        Task wrapper for PriorityQueue. Ordered by increasing task arrival time.
    """

    def __init__(self, task: Task):
        self.task = task
        self.priority = -task.log.task_arrival_at_scheduler_timestamp

    def __lt__(self, other):
        return self.priority < other.priority
    
    def __str__(self):
        return f"[PRIORITY: {self.priority}] {self.task}"
    
    def __repr__(self):
        return self.__str__()


def _flex_form_largest_batch(simulation, model_id: int, time: float) -> Batch:
    tasks = []
    skipped_tasks = []
    while simulation.model_queues[model_id].qsize() > 0:
        ot = simulation.model_queues[model_id].get()
        if time < ot.task.log.task_placed_on_worker_queue_timestamp:
            skipped_tasks.append(ot)
            continue
        tasks.append(ot.task)
        if len(tasks) == tasks[0].max_batch_size:
            break
    for ot in skipped_tasks:
        simulation.model_queues[model_id].put(ot)
    simulation._batch_counter += 1
    return Batch(simulation._batch_counter-1, tasks)


def flex_schedule_job_on_arrival(simulation, job: Job, current_time: float):
    # TODO: priority by SLO? task queues are FCFS
    for task in job.tasks:
        if len(task.required_task_ids) == 0:
            if task.model.model_id not in simulation.model_queues:
                simulation.model_queues[task.model.model_id] = PriorityQueue()
            simulation.model_queues[task.model.model_id].put(OrderedTask(task))
    return flex_schedule_tasks_on_arrival(simulation, current_time)


def flex_schedule_tasks_on_arrival(simulation, current_time: float, info_staleness=LOAD_INFORMATION_STALENESS):
    events = []
    for worker in simulation.workers:
        worker_models = worker.GPU_state.placed_models(max(0, current_time - info_staleness))
        for model in worker_models:
            curr_batch_size = 0
            if model.model_id in simulation.worker_states[worker.worker_id]:
                curr_batch = simulation.worker_states[worker.worker_id][model.model_id]
                curr_batch_size = curr_batch.size() if not curr_batch is None else 0

            if model.model_id not in simulation.model_queues:
                continue # no tasks queued

            model_queue = simulation.model_queues[model.model_id]
            if model_queue.qsize() == 0: # no tasks queued
                continue

            largest_batch_size = min(len([t for t in model_queue.queue if current_time >= t.task.log.task_placed_on_worker_queue_timestamp]),
                                     model_queue.queue[0].task.max_batch_size)
            if largest_batch_size == 0: # no tasks queued for [current_time]
                continue

            if curr_batch_size == 0:
                batch = _flex_form_largest_batch(simulation, model.model_id, current_time)
                simulation.worker_states[worker.worker_id][model.model_id] = batch
                events.append(EventOrders(
                    current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size), 
                    BatchArrivalAtWorker(simulation, worker, batch)))
            elif largest_batch_size >= FLEX_LAMBDA * curr_batch_size:
                batch = _flex_form_largest_batch(simulation, model.model_id, current_time)
                old_batch_id = simulation.worker_states[worker.worker_id][model.model_id].id
                simulation.worker_states[worker.worker_id][model.model_id] = batch
                events.append(EventOrders(
                    current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size), 
                    BatchPreemptionAtWorker(simulation, worker, batch, old_batch_id)))
    return events


def flex_schedule_on_batch_completion(simulation, worker: Worker, completed_batch: Batch, current_time: float, info_staleness=LOAD_INFORMATION_STALENESS):
    # clear scheduler worker state iff not assigned to a new batch already
    if simulation.worker_states[worker.worker_id][completed_batch.model.model_id].id == completed_batch.id:
        simulation.worker_states[worker.worker_id][completed_batch.model.model_id] = None
    
    largest_batch = (-1, 0) # (model_id, batch size)
    for model_id, task_queue in simulation.model_queues.items():
        if task_queue.qsize() == 0:
            continue

        # static allocation
        if all(m.model.model_id != model_id for m in worker.GPU_state.placed_model_states(max(0, current_time-info_staleness))):
            continue

        # already assigned
        if simulation.worker_states[worker.worker_id][model_id]:
            continue
        
        curr_largest_possible = min(len([t for t in task_queue.queue if current_time >= t.task.log.task_placed_on_worker_queue_timestamp]),
                                    task_queue.queue[0].task.max_batch_size)
        if curr_largest_possible == 0:
            continue
        elif curr_largest_possible > largest_batch[1]:
            largest_batch = (model_id, curr_largest_possible)
    
    if largest_batch[1] > 0:
        batch = _flex_form_largest_batch(simulation, largest_batch[0], current_time)
        simulation.worker_states[worker.worker_id][batch.model.model_id] = batch
        return [EventOrders(
            current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size), 
            BatchArrivalAtWorker(simulation, worker, batch))]
    
    return []